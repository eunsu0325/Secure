"""
ProjectionHead for COCONUT.

Reduces CCNet's 2048-D feature output to a lower dimensionality before
ProxyAnchor / NCM consume it. Designed for the continual learning regime
(few samples per class) where high-D prototype estimation is noise-dominated
and curse-of-dimensionality degrades open-set rejection.

Design:
- Pure linear projection (no MLP) — minimises overfitting risk under
  9-samples-per-user data regime.
- No bias term — projection lives on the unit hypersphere downstream
  (NCM/ProxyAnchor both L2-normalise their inputs).
- Initialised via PCA fit on a sample of pretrained CCNet features so
  the starting weights preserve the top-k variance directions
  (~99% variance retention for k ≥ 256 in typical face/palmprint
  embeddings) instead of a random direction.
- Remains trainable so ProxyAnchor can rotate the basis toward
  class-discriminative directions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection 2048-D → ``out_dim`` with PCA-friendly initialisation.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality (CCNet output = 2048).
    out_dim : int
        Output dimensionality after projection.

    Notes
    -----
    The weight tensor has shape ``(out_dim, in_dim)``. ``init_with_pca``
    populates it with the top-``out_dim`` right singular vectors of a
    sample of CCNet features so that at training start the projection is
    equivalent to PCA of pretrained features.
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        if out_dim > in_dim:
            raise ValueError(
                f"ProjectionHead: out_dim ({out_dim}) must be <= in_dim ({in_dim})"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        # bias=False keeps the projection a pure linear map; the L2
        # normalisation in NCM/ProxyAnchor removes the need for a bias term.
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # Identity-like default (top out_dim coords pass through). Replaced
        # by init_with_pca before training in the normal flow; kept as a
        # safe fallback if PCA fit is skipped.
        with torch.no_grad():
            self.proj.weight.zero_()
            eye_k = min(out_dim, in_dim)
            self.proj.weight[:eye_k, :eye_k] = torch.eye(eye_k)
        self._pca_initialised = False

    @torch.no_grad()
    def init_with_pca(self, features: torch.Tensor) -> dict:
        """Initialise ``self.proj.weight`` with the top-``out_dim`` PCA components.

        Parameters
        ----------
        features : Tensor
            Shape ``(N, in_dim)``. CCNet features sampled from in-domain
            training data (typically enroll_file). Must satisfy
            ``N >= out_dim + 1`` for the top-``out_dim`` components to be
            non-trivial; if not, the PCA still runs but the retained
            variance ratio will be reported and the caller can decide.

        Returns
        -------
        info : dict
            Diagnostic information:
              - ``n_samples``: N
              - ``retained_variance_ratio``: fraction of variance captured
                by the top out_dim components
              - ``rank_limit``: ``min(N - 1, in_dim)`` — the maximum number
                of meaningful components for this sample size.
        """
        if features.dim() != 2 or features.size(1) != self.in_dim:
            raise ValueError(
                f"ProjectionHead.init_with_pca expected shape (N, {self.in_dim}); "
                f"got {tuple(features.shape)}"
            )

        n_samples = features.size(0)
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean

        # Truncated SVD: works even when n_samples < in_dim (just gives
        # min(n_samples-1, in_dim) non-zero singular values).
        # torch.linalg.svd with full_matrices=False returns Vt of shape
        # (min(N, in_dim), in_dim) — exactly the principal directions we want.
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # Top out_dim principal components (rows of Vt are sorted by variance)
        top_k = Vt[:self.out_dim]  # (out_dim, in_dim)

        # If sample size limits the rank, the trailing rows of top_k may be
        # ill-defined; pad with the identity fallback (already there from
        # __init__) for those rows. Conservative — only overwrite the rows
        # that have a meaningful singular value.
        rank_limit = min(n_samples - 1, self.in_dim)
        usable_k = min(self.out_dim, rank_limit)

        target = self.proj.weight.data
        target[:usable_k] = top_k[:usable_k].to(target.device, dtype=target.dtype)
        # rows usable_k..out_dim retain the identity-like default

        # Variance retained by the top usable_k components
        var_total = (S ** 2).sum().clamp_min(1e-12)
        var_retained = (S[:usable_k] ** 2).sum() / var_total

        self._pca_initialised = True

        return {
            'n_samples': int(n_samples),
            'rank_limit': int(rank_limit),
            'usable_components': int(usable_k),
            'retained_variance_ratio': float(var_retained.item()),
            'singular_values_top5': S[:5].tolist(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear projection. Input ``(B, in_dim)`` → ``(B, out_dim)``."""
        return self.proj(x)


class ProjectionWrappedModel(nn.Module):
    """Wraps a CCNet backbone so that every feature-extraction call also passes
    through a :class:`ProjectionHead`.

    Why a wrapper instead of monkey-patching?
        Monkey-patching ``model.forward`` / ``model.getFeatureCode`` on the
        original CCNet instance does work for direct callers but can interact
        badly with framework internals (e.g. torch script tracing, deepcopy,
        `.to()` device moves performed before patching, state_dict/load_state_dict
        round-trips). A proper nn.Module wrapper is the canonical solution and
        composes correctly with every downstream code path that reads
        ``trainer.model``.

    What gets registered:
        - ``self.ccnet`` — registered as a submodule (so .train()/.eval()/.to()
          all propagate and `ccnet.parameters()` works as expected).
        - ``self.projection`` — also registered as a submodule.

    The trainer constructs the optimiser explicitly with
    ``wrapper.ccnet.parameters()`` and ``wrapper.projection.parameters()`` so
    the two groups can get separate learning rates without parameter overlap.

    Forwarded methods:
        - ``forward(x)`` → ``projection(ccnet(x))``
        - ``getFeatureCode(x)`` → ``projection(ccnet.getFeatureCode(x))``

    All other attributes (including custom CCNet attrs like
    ``_pretrained_load_info``) are delegated to ``self.ccnet`` via
    ``__getattr__`` so external code that expects "the CCNet model" still
    works transparently.
    """

    def __init__(self, ccnet: nn.Module, projection: 'ProjectionHead'):
        super().__init__()
        self.ccnet = ccnet
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.ccnet(x))

    def getFeatureCode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.ccnet.getFeatureCode(x))

    def __getattr__(self, name: str):
        # First let nn.Module find submodules / parameters / buffers.
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Fall back to attributes set directly on the wrapped ccnet
            # (e.g. _pretrained_load_info that train_coconut.py reads).
            ccnet = self.__dict__.get('_modules', {}).get('ccnet')
            if ccnet is not None and hasattr(ccnet, name):
                return getattr(ccnet, name)
            raise
