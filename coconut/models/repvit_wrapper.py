"""RepViT wrapper for the Exp1 robustness backbone.

RepViT is used in Exp1 as a CCNet-artifact defence: the same A/B/C protocol
is run with a non-palmprint-specialised backbone to confirm the protocol gap
is not an artefact of CCNet specifically. Absolute numbers from RepViT are
not the claim; the claim is that the closed-set / static / sequential gap
qualitatively persists.

Hard rules:
  * No fine-tuning. Parameters are frozen at ``__init__`` and a runner-side
    assert defends against regressions.
  * timm's ``train()``/``eval()`` are NOT overridden. PyTorch ``model.eval()``
    dispatches to ``self.train(False)`` internally, so any override breaks
    framework conventions and timm's own internal calls.
  * ImageNet normalization happens **exactly once** inside ``getFeatureCode``,
    not in the data transform. This protects against double-normalization if
    cached embeddings are accidentally reused under a different transform.
  * If timm's pretrained download fails (offline / hash mismatch / unknown
    model name), the wrapper raises a clear error instructing the user to
    provide ``checkpoint_path``. Random-initialised weights are never
    silently used in Exp1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RepViTWrapper(nn.Module):
    """Wraps a timm RepViT model behind the ``getFeatureCode`` contract.

    The Exp1 score-extraction path (``coconut.openset.score_extraction``) calls
    ``model.getFeatureCode(x)`` directly. CCNet exposes this method natively;
    timm models do not, so this wrapper provides it.

    Args:
        model_name: timm identifier; default ``repvit_m1_0.dist_450e_in1k``.
            ``repvit_m0_9.dist_450e_in1k`` is allowed via config for fast
            iteration.
        pretrained: only used when ``checkpoint_path`` is None. Ignored
            otherwise.
        checkpoint_path: optional local weights file. When set, takes
            precedence over the timm download path.
        l2_normalize: if True (default), the returned feature is L2-normalized
            on the last dim. CCNet's pipeline expects unnormalized features
            and lets NCM normalize internally; for RepViT we normalize at
            wrapper boundary because timm checkpoints come with arbitrary
            magnitude scales that shift threshold ranges run-to-run.
    """

    def __init__(
        self,
        model_name: str = "repvit_m1_0.dist_450e_in1k",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover -- env error path
            raise ImportError(
                "timm is required for the RepViT backbone. "
                "Install it with `pip install timm>=0.9`."
            ) from exc

        self.model_name = str(model_name)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self._pretrained_requested = bool(pretrained)

        if self.checkpoint_path:
            try:
                backbone = timm.create_model(
                    self.model_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"timm.create_model failed for model_name='{self.model_name}'. "
                    "Check that the model name is valid in your timm version."
                ) from exc

            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"RepViT checkpoint_path does not exist: {ckpt_path}"
                )
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            stripped = { (k[len("module."):] if k.startswith("module.") else k): v
                         for k, v in state.items() }
            missing, unexpected = backbone.load_state_dict(stripped, strict=False)
            if missing:
                print(f"[RepViT] checkpoint missing {len(missing)} key(s); first: {list(missing)[:5]}")
            if unexpected:
                print(f"[RepViT] checkpoint had {len(unexpected)} unexpected key(s); first: {list(unexpected)[:5]}")
            self._weights_source = str(ckpt_path.resolve())
        else:
            try:
                backbone = timm.create_model(
                    self.model_name,
                    pretrained=self._pretrained_requested,
                    num_classes=0,
                    global_pool="avg",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"RepViT pretrained download failed for model_name='{self.model_name}'; "
                    "provide model.checkpoint_path in the config to use a local copy. "
                    f"Underlying error: {exc!r}"
                ) from exc
            self._weights_source = f"timm:{self.model_name}"

        self.backbone = backbone
        self.l2_normalize = bool(l2_normalize)

        # Probe feature dim with a no-grad forward on a dummy input. This also
        # ensures the model is functional before the runner wires it up.
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self._forward_features(dummy)
            self.feature_dim = int(feat.shape[-1])

        for p in self.parameters():
            p.requires_grad_(False)

        # Register normalization constants as buffers so .to(device) moves them.
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean, persistent=False)
        self.register_buffer("_imagenet_std", std, persistent=False)

    # NB: train() and eval() are intentionally NOT overridden.

    @property
    def weights_source(self) -> str:
        """Identifier of where the weights came from; used in embedding_meta."""
        return self._weights_source

    @property
    def transform_name(self) -> str:
        return f"repvit_eval_3x224"

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def getFeatureCode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features for downstream NCM scoring.

        ``x`` may be 1-channel (will be repeated to 3) or already 3-channel.
        Input is expected in [0, 1] range (i.e. post-``ToTensor``); ImageNet
        normalization is applied here exactly once.
        """
        if x.dim() != 4:
            raise ValueError(f"RepViT expects 4D input, got shape {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"RepViT expects 1 or 3 channels, got {x.shape[1]}")
        x = (x - self._imagenet_mean) / self._imagenet_std
        feat = self._forward_features(x)
        if self.l2_normalize:
            feat = F.normalize(feat, p=2, dim=-1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # alias
        return self.getFeatureCode(x)
