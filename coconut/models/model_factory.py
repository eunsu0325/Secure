"""Backbone factory for Exp1.

Single entry point ``build_backbone(cfg, device, project_root)`` that returns
the model already on-device, in eval mode, with all parameters
``requires_grad=False``. Also returns the matching transform pipeline and the
backbone's output feature dim.

This is the only place that knows the difference between CCNet and RepViT.
The runner and the dataset builders work against the common contract:

    model.getFeatureCode(x) -> Tensor of shape (B, feature_dim)

Hard rules:
  * Exp1 backbones are never fine-tuned. The factory freezes parameters before
    returning. The runner additionally asserts no parameter has
    ``requires_grad=True``.
  * Tongji + CCNet leakage guard: if ``cfg.dataset.name == 'tongji_roi'`` and
    architecture is ``ccnet`` and the resolved pretrained path basename matches
    ``tongji*.pth``, the factory raises. Tongji-on-CCNet must use a different
    checkpoint or a different backbone.
  * Relative paths in the config (``checkpoints/tongji.pth``,
    ``checkpoints/repvit_m1_0.pth``) are resolved against ``project_root``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from coconut.models.ccnet import ccnet
from coconut.models.pretrained_loader import PretrainedLoader
from coconut.models.repvit_wrapper import RepViTWrapper
from coconut.data.transforms import get_scr_transforms, get_repvit_transforms


def _resolve_path(p: Optional[str], project_root: Path) -> Optional[Path]:
    if p is None or p == "":
        return None
    path = Path(p)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path


def _check_tongji_leakage(
    dataset_name: str,
    architecture: str,
    resolved_pretrained: Optional[Path],
) -> None:
    if architecture != "ccnet":
        return
    if dataset_name != "tongji_roi":
        return
    if resolved_pretrained is None:
        return
    if resolved_pretrained.name.lower().startswith("tongji"):
        raise RuntimeError(
            "Tongji-on-CCNet leakage guard: dataset_name='tongji_roi' with "
            f"CCNet pretrained '{resolved_pretrained.name}' is forbidden in "
            "Exp1 because CCNet was pretrained on Tongji. Use a different "
            "checkpoint (e.g. an IITD-pretrained CCNet) or switch the "
            "backbone to repvit."
        )


class _BackboneBundle:
    """Lightweight bundle for the runner."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable,
        feature_dim: int,
        architecture: str,
        model_name: Optional[str],
        weights_source: str,
        transform_name: str,
        input_height: int,
        input_width: int,
        channels: int,
    ) -> None:
        self.model = model
        self.transform = transform
        self.feature_dim = int(feature_dim)
        self.architecture = str(architecture)
        self.model_name = model_name
        self.weights_source = str(weights_source)
        self.transform_name = str(transform_name)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.channels = int(channels)


def build_backbone(
    cfg: dict,
    device: torch.device,
    project_root: Path,
) -> Tuple[nn.Module, Callable, int]:
    """Construct the backbone described by ``cfg['model']``.

    Returns ``(model, transform, feature_dim)`` for callers that just want
    that triple. The model is already on ``device``, in eval mode, fully
    frozen.

    The richer metadata (architecture, model_name, weights_source,
    transform_name, input H/W/C) is also available via
    ``build_backbone_bundle`` for runners that need to populate
    ``embedding_meta``.
    """
    bundle = build_backbone_bundle(cfg, device, project_root)
    return bundle.model, bundle.transform, bundle.feature_dim


def build_backbone_bundle(
    cfg: dict,
    device: torch.device,
    project_root: Path,
) -> _BackboneBundle:
    project_root = Path(project_root)
    model_cfg = cfg.get("model", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}
    architecture = str(model_cfg.get("architecture", "")).lower()
    dataset_name = str(dataset_cfg.get("name", ""))
    height = int(dataset_cfg.get("height", 0))
    width = int(dataset_cfg.get("width", 0))
    channels = int(dataset_cfg.get("channels", 0))

    if architecture == "ccnet":
        weight = float(model_cfg.get("competition_weight", 0.8))
        resolved = _resolve_path(model_cfg.get("pretrained_path"), project_root)
        _check_tongji_leakage(dataset_name, architecture, resolved)
        if resolved is None:
            raise ValueError("ccnet config requires model.pretrained_path")

        model = ccnet(weight=weight)
        model = PretrainedLoader.load_ccnet_pretrained(
            model, resolved, device=str(device), verbose=True
        )
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Probe feature dim from CCNet (existing CCNet returns 2048).
        with torch.no_grad():
            dummy = torch.zeros(1, max(channels, 1), max(height, 128), max(width, 128), device=device)
            feature_dim = int(model.getFeatureCode(dummy).shape[-1])

        transform = get_scr_transforms(
            train=False,
            imside=height if height > 0 else 128,
            channels=channels if channels > 0 else 1,
        )
        bundle = _BackboneBundle(
            model=model,
            transform=transform,
            feature_dim=feature_dim,
            architecture="ccnet",
            model_name=None,
            weights_source=str(resolved.resolve()),
            transform_name=f"scr_eval_{channels or 1}x{height or 128}",
            input_height=height or 128,
            input_width=width or 128,
            channels=channels or 1,
        )
        return bundle

    if architecture == "repvit":
        repvit_model_name = str(model_cfg.get("model_name", "repvit_m1_0.dist_450e_in1k"))
        pretrained = bool(model_cfg.get("pretrained", True))
        ckpt = model_cfg.get("checkpoint_path")
        resolved_ckpt = _resolve_path(ckpt, project_root) if ckpt else None
        if resolved_ckpt is not None and not resolved_ckpt.is_file():
            raise FileNotFoundError(
                f"RepViT checkpoint_path resolved to non-existent file: {resolved_ckpt}"
            )

        wrapper = RepViTWrapper(
            model_name=repvit_model_name,
            pretrained=pretrained,
            checkpoint_path=str(resolved_ckpt) if resolved_ckpt else None,
        )
        wrapper.to(device)
        wrapper.eval()
        for p in wrapper.parameters():
            p.requires_grad_(False)

        feature_dim = int(wrapper.feature_dim)

        transform = get_repvit_transforms(
            train=False,
            imside=height if height > 0 else 224,
        )
        bundle = _BackboneBundle(
            model=wrapper,
            transform=transform,
            feature_dim=feature_dim,
            architecture="repvit",
            model_name=repvit_model_name,
            weights_source=wrapper.weights_source,
            transform_name=wrapper.transform_name,
            input_height=height or 224,
            input_width=width or 224,
            channels=channels or 3,
        )
        return bundle

    raise ValueError(
        f"Unknown architecture: {architecture!r}. Expected 'ccnet' or 'repvit'."
    )
