"""Embedding extraction for exp1_baselines (D1h: manifest-driven).

Extracts L2-normalized backbone embeddings for evaluation identities only.
Eval identities are the four canonical subject_splits — train_backbone is
NEVER extracted (its identities are never queried under any protocol).

  - base_anchor    : always-enrolled gallery palms
  - future         : sequentially enrolled palms (Protocol C trajectory)
  - external_dev   : τ-calibration unknown palms
  - external_test  : never-enrolled unknown palms

NPZ output schema (drop legacy `split_name`; use canonical fields)::

    embeddings    [N, D]  float32, L2-normalized
    palm_id       [N]     int64   (recognition class)
    subject_id    [N]     int64   (partitioning unit)
    identity_id   [N]     str     ("subject_<id>-a/-b" Tongji Tier 2, etc.)
    session_id    [N]     str     ("session1"/"session2"/"")
    phase_id      [N]     str     ("F"/"S"/"")
    sample_id     [N]     int64   (0..9 Tongji; per-dataset semantics)
    subject_split [N]     str     ("base_anchor"|"future"|"external_dev"|"external_test")
    sample_role   [N]     str     ("enroll"|"query"|"unused"|"candidate")
    path          [N]     str     (absolute image path, for traceability)

Usage::

    python exp1_baselines/extract_embeddings.py \\
        --model mfn \\
        --checkpoint exp1_baselines/checkpoints/mfn_arcface_tongji_scratch_112.pt \\
        --manifest_dir experiments/generated/tongji \\
        --output_npz exp1_baselines/embeddings/mfn_tongji_112_embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.backbones.mobilefacenet import MobileFaceNet
from exp1_baselines.backbones.iresnet import iresnet50
from exp1_baselines.backbones.ccnet import ccnet
from exp1_baselines.backbones.ccm_mfn import ccm_mfn
from exp1_baselines.datasets.manifest_io import get_eval_records, load_manifest
from exp1_baselines.datasets.tongji_dataset import (
    TongjiROIDataset,
    build_baseline_transform,
)


def build_backbone(
    architecture: str,
    embedding_dim: int,
    backbone_kwargs: dict | None = None,
) -> torch.nn.Module:
    arch = architecture.lower()
    backbone_kwargs = backbone_kwargs or {}
    if arch in {"mobilefacenet", "mfn"}:
        return MobileFaceNet(embedding_dim=embedding_dim)
    elif arch in {"iresnet50", "ir50"}:
        return iresnet50(num_features=embedding_dim)
    elif arch == "ccnet":
        net = ccnet(
            weight=float(backbone_kwargs.get("weight", 0.8)),
            use_dropout=bool(backbone_kwargs.get("use_dropout", False)),
        )
        if int(net.embedding_dim) != int(embedding_dim):
            raise ValueError(
                f"ccnet embedding_dim is fixed at {net.embedding_dim} "
                f"by the architecture; checkpoint requested {embedding_dim}."
            )
        return net
    elif arch in {"ccm_mfn", "ccm-mfn"}:
        # V19 Path 3b — see ccm_mfn.py docstring and plan §V19.
        return ccm_mfn(
            embedding_dim=embedding_dim,
            ccm_n_competitor=int(backbone_kwargs.get("ccm_n_competitor", 9)),
            ccm_ksize=int(backbone_kwargs.get("ccm_ksize", 7)),
            ccm_init_ratio=float(backbone_kwargs.get("ccm_init_ratio", 1.0)),
            ccm_weight=float(backbone_kwargs.get("ccm_weight", 0.8)),
            alpha_init=float(backbone_kwargs.get("alpha_init", 0.05)),
        )
    else:
        raise ValueError(f"unknown architecture: {architecture}")


def collect_eval_records(manifest_dir: Path) -> List[dict]:
    """Read manifest CSV and return one dict per evaluation row.

    Each dict carries every NPZ column needed downstream so the extraction
    loop can populate arrays without re-parsing image paths.
    """
    manifest_path = manifest_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest CSV not found at {manifest_path}. "
            "Run the appropriate `*_manifest.py` builder first."
        )
    df = load_manifest(manifest_path)
    eval_df = get_eval_records(df)
    if eval_df.empty:
        raise RuntimeError(
            f"no evaluation rows in {manifest_path} "
            "(expected base_anchor/future/external_dev/external_test)"
        )
    out: List[dict] = []
    for _, row in eval_df.iterrows():
        abs_path = row["image_path"]
        if not Path(abs_path).is_absolute():
            abs_path = str((PROJECT_ROOT / abs_path).resolve())
        out.append({
            "path": abs_path,
            "palm_id": int(row["palm_id"]),
            "subject_id": int(row["subject_id"]),
            "identity_id": str(row["identity_id"]),
            "session_id": str(row["session_id"]),
            "phase_id": str(row["phase_id"]),
            "sample_id": int(row["sample_id"]),
            "subject_split": str(row["subject_split"]),
            "sample_role": str(row["sample_role"]),
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract embeddings for exp1_baselines")
    parser.add_argument(
        "--model", required=True,
        choices=["mfn", "ir50", "ccnet", "ccm_mfn"],
        help=(
            "Architecture name. 'ccm_mfn' is V19 Path 3b's CCNet-inspired "
            "competitive-mechanism residual adapter on MFN; NOT a reproduction "
            "of CCNet or RegPalm."
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--manifest_dir", required=True, type=str)
    parser.add_argument("--output_npz", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--image_size", default=None, type=int,
                        help="If unset, auto-derive from checkpoint config.")
    parser.add_argument("--channels", default=None, type=int,
                        help="If unset, auto-derive from checkpoint config.")
    args = parser.parse_args(argv)

    if args.device:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[device] {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["config"]["model"]["architecture"]
    embedding_dim = int(ckpt.get("embedding_dim", 512))
    backbone_kwargs = ckpt["config"]["model"].get("backbone_kwargs", {}) or {}
    cfg_dataset = ckpt["config"].get("dataset", {})
    image_size = int(args.image_size) if args.image_size is not None \
        else int(cfg_dataset.get("image_size", 112))
    channels = int(args.channels) if args.channels is not None \
        else int(cfg_dataset.get("channels", 3))
    print(f"[ckpt] arch={arch}, embedding_dim={embedding_dim}, "
          f"image_size={image_size}, channels={channels}, "
          f"epochs_completed={ckpt.get('epochs_completed', '?')}")

    backbone = build_backbone(arch, embedding_dim, backbone_kwargs).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = (PROJECT_ROOT / manifest_dir).resolve()
    records = collect_eval_records(manifest_dir)
    n = len(records)
    print(f"[data] {n} evaluation images across "
          f"{len(set(r['subject_split'] for r in records))} subject_splits")

    transform = build_baseline_transform(image_size=image_size, channels=channels)
    # The dataset wrapper expects (path, label) pairs; we use palm_id as
    # the carried label (consumers don't rely on it — they read palm_id from NPZ).
    dataset = TongjiROIDataset(
        [(r["path"], r["palm_id"]) for r in records],
        transform=transform, image_size=image_size, channels=channels,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    all_emb = np.zeros((n, embedding_dim), dtype=np.float32)
    all_palm = np.zeros((n,), dtype=np.int64)
    all_subject = np.zeros((n,), dtype=np.int64)
    all_identity = np.empty((n,), dtype=object)
    all_session = np.empty((n,), dtype=object)
    all_phase = np.empty((n,), dtype=object)
    all_sample = np.zeros((n,), dtype=np.int64)
    all_split = np.empty((n,), dtype=object)
    all_role = np.empty((n,), dtype=object)
    all_paths = np.empty((n,), dtype=object)

    cursor = 0
    with torch.no_grad():
        for images, _labels, paths in loader:
            images = images.to(device, non_blocking=True)
            feats = backbone(images)
            feats = F.normalize(feats.float(), p=2, dim=1)
            bsz = feats.shape[0]
            all_emb[cursor: cursor + bsz] = feats.cpu().numpy()
            for i in range(bsz):
                idx = cursor + i
                rec = records[idx]
                # DataLoader doesn't shuffle; loader path must match record path.
                if str(paths[i]) != rec["path"]:
                    raise RuntimeError(
                        f"loader path mismatch at idx {idx}: "
                        f"{paths[i]} != {rec['path']}"
                    )
                all_palm[idx] = rec["palm_id"]
                all_subject[idx] = rec["subject_id"]
                all_identity[idx] = rec["identity_id"]
                all_session[idx] = rec["session_id"]
                all_phase[idx] = rec["phase_id"]
                all_sample[idx] = rec["sample_id"]
                all_split[idx] = rec["subject_split"]
                all_role[idx] = rec["sample_role"]
                all_paths[idx] = rec["path"]
            cursor += bsz

    if cursor != n:
        raise RuntimeError(f"extracted {cursor} images, expected {n}")

    norms = np.linalg.norm(all_emb, axis=1)
    print(f"[emb] L2-norm range: {norms.min():.5f} - {norms.max():.5f}")
    if not ((norms > 0.99).all() and (norms < 1.01).all()):
        raise RuntimeError("L2-norm sanity failed (embeddings not unit norm)")

    # ----- Plan §V17.11 extract-time telemetry -----
    import subprocess as _sp
    try:
        extraction_git_commit = _sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception:
        extraction_git_commit = "unknown"
    v17_11_extract_metadata = {
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "embedding_min": float(all_emb.min()),
        "embedding_max": float(all_emb.max()),
        "extraction_device": str(device),
        "torch_version": torch.__version__,
        "n_samples_extracted": int(all_emb.shape[0]),
        "embedding_dim": int(all_emb.shape[1]),
        "extraction_git_commit": extraction_git_commit,
        "ckpt_path": str(ckpt_path),
        "ckpt_arch": str(arch),
        "ckpt_image_size": int(image_size),
        "ckpt_channels": int(channels),
        "ckpt_v17_11_metadata_present": ("v17_11_metadata" in ckpt),
        "spec_version": "V17.11",
    }

    out_npz = Path(args.output_npz)
    if not out_npz.is_absolute():
        out_npz = PROJECT_ROOT / out_npz
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=all_emb,
        palm_id=all_palm,
        subject_id=all_subject,
        identity_id=all_identity,
        session_id=all_session,
        phase_id=all_phase,
        sample_id=all_sample,
        subject_split=all_split,
        sample_role=all_role,
        path=all_paths,
        v17_11_extract_metadata=np.array(
            json.dumps(v17_11_extract_metadata), dtype=object,
        ),
    )
    print(f"[save] {out_npz} ({all_emb.shape})")

    # Sidecar JSON for human/script inspection
    metadata_json_path = out_npz.with_suffix(".v17_extract_metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(v17_11_extract_metadata, f, indent=2)
    print(f"[save] V17.11 extract-time metadata sidecar -> {metadata_json_path}")

    from collections import Counter
    print(f"[counts] subject_split: {dict(Counter(all_split.tolist()))}")
    print(f"[counts] sample_role:   {dict(Counter(all_role.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
