"""Generate per-(dataset, backbone) YAML configs for exp1_run.py.

Usage from a builder::

    from experiments.exp1_common import config_factory

    config_factory.write_configs(
        out_dir=Path("experiments/generated/bjtu_v2"),
        dataset_name="bjtu_v2",
        future_batch_size=10,
        static_gallery_sizes=[30, 60, 100, 150],
        backbones=("ccnet", "repvit"),
    )

The emitted YAML matches the schema described in the plan §5 and the keys
``exp1_run.py`` consumes: precomputed_identity_split, precomputed_sample_split,
metadata_file, height/width/channels, model.architecture-specific fields,
scoring.target_fpir + additional_fpirs, protocol_c.future_batch_size,
static_gallery_sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml


_GENERATED_REL = "experiments/generated"
_RESULTS_REL = "experiments/results"


def _build_dataset_block(dataset_name: str, height: int, width: int, channels: int) -> dict:
    return {
        "name": dataset_name,
        "txt_file": f"{_GENERATED_REL}/{dataset_name}/all.txt",
        "precomputed_identity_split": f"{_GENERATED_REL}/{dataset_name}/identity_split.json",
        "precomputed_sample_split": f"{_GENERATED_REL}/{dataset_name}/sample_split.json",
        "metadata_file": f"{_GENERATED_REL}/{dataset_name}/metadata.json",
        "base_path": "",
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
    }


def _build_ccnet_model_block(
    competition_weight: float,
    pretrained_path: str,
) -> dict:
    return {
        "architecture": "ccnet",
        "competition_weight": float(competition_weight),
        "pretrained_path": str(pretrained_path),
    }


def _build_repvit_model_block(
    model_name: str,
    pretrained: bool,
    checkpoint_path: Optional[str],
) -> dict:
    return {
        "architecture": "repvit",
        "model_name": str(model_name),
        "pretrained": bool(pretrained),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def write_configs(
    out_dir: Path,
    *,
    dataset_name: str,
    future_batch_size: int,
    static_gallery_sizes: Sequence[int],
    backbones: Iterable[str] = ("ccnet", "repvit"),
    seed: int = 42,
    target_fpir: float = 0.01,
    additional_fpirs: Sequence[float] = (0.05,),
    # CCNet inputs
    ccnet_height: int = 128,
    ccnet_width: int = 128,
    ccnet_channels: int = 1,
    ccnet_competition_weight: float = 0.8,
    ccnet_pretrained_path: str = "checkpoints/tongji.pth",
    # RepViT inputs
    repvit_height: int = 224,
    repvit_width: int = 224,
    repvit_channels: int = 3,
    repvit_model_name: str = "repvit_m1_0.dist_450e_in1k",
    repvit_pretrained: bool = True,
    repvit_checkpoint_path: Optional[str] = None,
    # Compatibility blocks the runner still reads
    n_enroll: int = 2,
    n_dev: int = 1,
    n_test_min: int = 2,
    n_base: int = 0,
    n_future: int = 0,
    n_external: int = 0,
) -> List[Path]:
    """Write one YAML per backbone into ``out_dir``.

    Returns the list of paths written. ``ccnet`` always uses 1×128×128;
    ``repvit`` always uses 3×224×224. Other parameters mirror the dataset
    builder's choices.

    The legacy top-level ``identity_split`` and ``sample_split`` blocks are
    still emitted so back-compat with the runner's pre-precomputed code path
    is preserved; when ``precomputed_*`` fields are present the runner ignores
    those counts.
    """
    out_dir = Path(out_dir)
    written: List[Path] = []
    for backbone in backbones:
        backbone = backbone.lower()
        if backbone == "ccnet":
            dataset_block = _build_dataset_block(
                dataset_name, ccnet_height, ccnet_width, ccnet_channels
            )
            model_block = _build_ccnet_model_block(
                ccnet_competition_weight, ccnet_pretrained_path
            )
        elif backbone == "repvit":
            dataset_block = _build_dataset_block(
                dataset_name, repvit_height, repvit_width, repvit_channels
            )
            model_block = _build_repvit_model_block(
                repvit_model_name, repvit_pretrained, repvit_checkpoint_path
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        payload = {
            "experiment": {
                "name": f"exp1_{dataset_name}_{backbone}",
                "seed": int(seed),
                "output_dir": f"{_RESULTS_REL}/{dataset_name}_{backbone}",
            },
            "dataset": dataset_block,
            "identity_split": {
                "n_base": int(n_base),
                "n_future": int(n_future),
                "n_external": int(n_external),
            },
            "sample_split": {
                "n_enroll": int(n_enroll),
                "n_dev": int(n_dev),
                "n_test_min": int(n_test_min),
            },
            "model": model_block,
            "scoring": {
                "mode": "cosine",
                "target_fpir": float(target_fpir),
                "additional_fpirs": [float(x) for x in additional_fpirs],
                "run_snorm_recalib": True,
            },
            "protocol_c": {
                "future_batch_size": int(future_batch_size),
            },
            "static_gallery_sizes": [int(x) for x in static_gallery_sizes],
        }

        path = out_dir / f"exp1_{dataset_name}_{backbone}.yaml"
        _write_yaml(path, payload)
        written.append(path)
    return written
