"""ArcFace backbone training script for exp1_baselines.

Trains MobileFaceNet or IR-ResNet50 from scratch on Tongji ROI base_ids
(320 identities, session1+session2, 6,400 images) with ArcFace classification head.

After training, the ArcFace head is discarded and only the backbone is kept
as feature extractor.

Usage::

    python exp1_baselines/train_arcface.py \\
        --config exp1_baselines/configs/mfn_arcface_tongji_112.yaml

    python exp1_baselines/train_arcface.py \\
        --config exp1_baselines/configs/ir50_arcface_tongji_112.yaml \\
        --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.backbones.mobilefacenet import MobileFaceNet
from exp1_baselines.backbones.iresnet import iresnet50
from exp1_baselines.backbones.ccnet import ccnet
from exp1_baselines.losses.arcface import ArcFaceHead
from exp1_baselines.datasets.tongji_dataset import (
    TongjiROIDataset,
    PKBatchSampler,
    build_baseline_transform,
)
from exp1_baselines.datasets.manifest_io import (
    load_manifest,
    get_train_records,
    get_val_ckpt_records,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def build_backbone(
    architecture: str,
    embedding_dim: int,
    backbone_kwargs: Optional[Dict] = None,
) -> nn.Module:
    """Construct a backbone module by architecture name.

    Args:
        architecture: one of {mobilefacenet, mfn, iresnet50, ir50, ccnet}.
        embedding_dim: feature dim. For ccnet this is fixed at 6144 by the
            architecture (verified internally; mismatch raises).
        backbone_kwargs: optional architecture-specific overrides
            (e.g., {"weight": 0.8, "use_dropout": false} for ccnet).
    """
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
                f"by the architecture; config requested {embedding_dim}. "
                f"Set embedding_dim: {net.embedding_dim} in the config."
            )
        return net
    else:
        raise ValueError(f"unknown architecture: {architecture}")


def _records_from_df(df, label_remap: Dict[int, int]) -> Tuple[List[Tuple[str, int]], List[int]]:
    """Convert a manifest sub-DataFrame into (records, labels) with contiguous labels."""
    records: List[Tuple[str, int]] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        abs_path = row["image_path"]
        if not Path(abs_path).is_absolute():
            abs_path = str((PROJECT_ROOT / abs_path).resolve())
        contiguous = label_remap[int(row["palm_id"])]
        records.append((abs_path, contiguous))
        labels.append(contiguous)
    return records, labels


def load_train_records(
    manifest_dir: Path,
) -> Tuple[List[Tuple[str, int]], List[int], int, List[Tuple[str, int]]]:
    """Load training + val_ckpt records from the canonical manifest CSV.

    Reads `manifest_dir/manifest.csv` (5-way split, sample_role-based) and
    returns:
      train_records:  (abs_path, contiguous_label) for subject_split=train_backbone
                      AND sample_role=train (image-level training set).
      train_labels:   parallel int labels (drives PKBatchSampler).
      num_classes:    number of unique palm_ids in train_backbone (recognition
                      classes for the ArcFace head — palm-level, not subject-level).
      val_ckpt_records: (abs_path, contiguous_label) for sample_role=val_ckpt
                      under the SAME label_remap; consumers may use these for
                      checkpoint selection (image-level holdout per plan D1g).

    Per plan: base_anchor / future / external_dev / external_test are NEVER
    used during backbone training — identity-disjoint from the eval pool.
    """
    manifest_path = manifest_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest CSV not found at {manifest_path}. "
            f"Run `python -m exp1_baselines.datasets.tongji_manifest` first."
        )
    df = load_manifest(manifest_path)

    train_df = get_train_records(df)
    if train_df.empty:
        raise RuntimeError(
            f"no training rows in {manifest_path} "
            "(expected subject_split=train_backbone & sample_role=train)"
        )

    palm_ids_sorted = sorted(int(x) for x in train_df["palm_id"].unique())
    label_remap: Dict[int, int] = {pid: idx for idx, pid in enumerate(palm_ids_sorted)}

    train_records, train_labels = _records_from_df(train_df, label_remap)

    val_df = get_val_ckpt_records(df)
    val_records: List[Tuple[str, int]] = []
    if not val_df.empty:
        # val_ckpt uses the SAME palm_ids as training (image-level holdout)
        unknown_palms = sorted(set(int(x) for x in val_df["palm_id"].unique()) - set(palm_ids_sorted))
        if unknown_palms:
            raise RuntimeError(
                f"val_ckpt rows have palm_ids not in train: {unknown_palms[:5]}... "
                "(expected image-level holdout — same palms as train_backbone&train)"
            )
        val_records, _ = _records_from_df(val_df, label_remap)

    return train_records, train_labels, len(palm_ids_sorted), val_records


def cosine_warmup_lr(epoch: int, total_epochs: int, warmup_epochs: int,
                     base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ArcFace backbone training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--manifest_dir", default=None, type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--amp", default=None, type=lambda v: v.lower() == "true")
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output", default=None, type=str,
                        help="override checkpoint output path")
    parser.add_argument("--smoke", action="store_true",
                        help="run only 1 batch + 1 epoch for sanity check")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.manifest_dir:
        cfg["dataset"]["manifest_dir"] = args.manifest_dir
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp
    if args.output:
        cfg["output"]["checkpoint_path"] = args.output

    seed = int(cfg["train"]["seed"])
    set_all_seeds(seed)

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
    print(f"[git_commit] {get_git_commit()}")

    manifest_dir = Path(cfg["dataset"]["manifest_dir"])
    if not manifest_dir.is_absolute():
        manifest_dir = (PROJECT_ROOT / manifest_dir).resolve()
    records, labels, num_classes, val_records = load_train_records(manifest_dir)
    print(
        f"[data] training images: {len(records)}, classes: {num_classes}, "
        f"val_ckpt images: {len(val_records)}"
    )
    if args.smoke:
        records = records[: cfg["train"]["batch_size"] * 4]
        labels = labels[: cfg["train"]["batch_size"] * 4]
        print(f"[smoke] truncated to {len(records)} images")

    image_size = int(cfg["dataset"]["image_size"])
    channels = int(cfg["dataset"].get("channels", 3))
    transform = build_baseline_transform(image_size=image_size, channels=channels)
    dataset = TongjiROIDataset(
        records, transform=transform, image_size=image_size, channels=channels,
    )

    P = int(cfg["train"]["P"])
    K = int(cfg["train"]["K"])
    batch_size = int(cfg["train"]["batch_size"])
    if P * K != batch_size:
        # adjust K to fit
        K = max(1, batch_size // P)
        print(
            f"[sampler] adjusting K to {K} so that P*K={P*K} matches "
            f"batch_size={batch_size}"
        )
    steps_per_epoch = max(1, int(math.ceil(len(records) / max(1, P * K))))
    sampler = PKBatchSampler(
        labels=labels, P=P, K=K, num_batches=steps_per_epoch, seed=seed
    )
    print(
        f"[sampler] PxK class-balanced; P={P}, K={K}, "
        f"batch_size={P*K}, steps_per_epoch={steps_per_epoch}"
    )

    num_workers = int(cfg["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embedding_dim = int(cfg["model"]["embedding_dim"])
    backbone_kwargs = cfg["model"].get("backbone_kwargs", {}) or {}
    backbone = build_backbone(
        cfg["model"]["architecture"], embedding_dim, backbone_kwargs,
    ).to(device)
    arcface_head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=float(cfg["arcface"]["margin"]),
        scale=float(cfg["arcface"]["scale"]),
        eps=float(cfg["arcface"]["eps"]),
    ).to(device)

    params = list(backbone.parameters()) + list(arcface_head.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"]["momentum"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["train"]["warmup_epochs"])
    base_lr = float(cfg["train"]["lr"])
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.smoke:
        epochs = 1

    log_interval = int(cfg["train"].get("log_interval", 10))

    log_records: List[Dict] = []
    out_ckpt = Path(cfg["output"]["checkpoint_path"])
    if not out_ckpt.is_absolute():
        out_ckpt = PROJECT_ROOT / out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["output"]["log_dir"])
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] epochs={epochs}, lr={base_lr}, amp={use_amp}")
    backbone.train()
    arcface_head.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_lr = cosine_warmup_lr(epoch, epochs, warmup_epochs, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = epoch_lr

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()
        for step, batch in enumerate(loader):
            images, batch_labels, _paths = batch
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                features = backbone(images)
            # ArcFace runs in FP32 for numerical safety
            logits = arcface_head(features.float(), batch_labels)
            loss = F.cross_entropy(logits, batch_labels)

            if not torch.isfinite(loss):
                print(f"[ERR] non-finite loss at epoch {epoch} step {step}: {loss.item()}")
                return 2

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                epoch_correct += int((preds == batch_labels).sum().item())
                epoch_total += int(batch_labels.numel())
                epoch_loss_sum += float(loss.item()) * batch_labels.numel()

            global_step += 1
            if (step + 1) % log_interval == 0 or step == 0:
                cur_acc = epoch_correct / max(1, epoch_total)
                print(
                    f"[ep {epoch+1}/{epochs} step {step+1}/{steps_per_epoch}] "
                    f"loss={loss.item():.4f} acc={cur_acc:.4f} lr={epoch_lr:.5f}"
                )
            if args.smoke and step >= 1:
                break

        avg_loss = epoch_loss_sum / max(1, epoch_total)
        avg_acc = epoch_correct / max(1, epoch_total)
        epoch_time = time.time() - epoch_start
        log_records.append({
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "loss": avg_loss,
            "acc": avg_acc,
            "time_sec": epoch_time,
        })
        print(
            f"[ep {epoch+1}] avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f} "
            f"time={epoch_time:.1f}s"
        )

    # Save checkpoint: backbone state_dict + meta. ArcFace head is discarded
    # but we record its config for reproducibility.
    checkpoint = {
        "backbone_state_dict": backbone.state_dict(),
        "arcface_head_state_dict": arcface_head.state_dict(),  # for reproducibility
        "config": cfg,
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "git_commit": get_git_commit(),
        "epochs_completed": epochs,
    }
    torch.save(checkpoint, out_ckpt)
    print(f"[save] checkpoint -> {out_ckpt}")

    with open(log_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)
    print(f"[save] train log -> {log_dir/'train_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
