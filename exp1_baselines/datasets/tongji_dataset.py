"""Tongji ROI dataset for exp1_baselines.

Image transform pipeline (per-backbone):
  MFN-ArcFace / IR50:
    1. Load BMP (grayscale or RGB)
    2. Convert to 3-channel RGB
    3. Resize to 112x112
    4. ToTensor (uint8 -> float in [0,1])
    5. Normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] -> [-1, 1]
  CCNet-ArcFace:
    1. Load BMP (grayscale or RGB)
    2. Convert to 1-channel grayscale (PIL "L" mode, ITU-R 601-2 luma)
    3. Resize to 128x128 (CCNet author native; FC layer is hardcoded for 128)
    4. ToTensor
    5. Normalize with mean=[0.5], std=[0.5] -> [-1, 1] (same dynamics as MFN)

Both backbones see the same underlying grayscale palmprint information; MFN
replicates gray to 3 channels (no info added) while CCNet keeps it as 1
channel. The [-1,1] value range is shared.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


def build_baseline_transform(
    image_size: int = 112,
    channels: int = 3,
) -> transforms.Compose:
    """Return the per-backbone preprocessing transform.

    Args:
        image_size: spatial size after Resize. 112 for MFN/IR50, 128 for CCNet.
        channels: number of input channels (1 for CCNet, 3 for MFN/IR50).
    """
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    norm_mean = [0.5] * channels
    norm_std = [0.5] * channels
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


class TongjiROIDataset(Dataset):
    """Generic ROI dataset reading (image_path, label) pairs.

    Args:
        records: list of (absolute_path, label) tuples.
        transform: torchvision transform applied to each loaded image.
        image_size: passed to default transform when transform is None.
        channels: 1 (grayscale, for CCNet) or 3 (RGB, for MFN/IR50).
    """

    def __init__(
        self,
        records: Sequence[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 112,
        channels: int = 3,
    ) -> None:
        self.records: List[Tuple[str, int]] = [
            (str(p), int(l)) for p, l in records
        ]
        self.channels = int(channels)
        if transform is None:
            transform = build_baseline_transform(image_size, channels=self.channels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: str) -> Image.Image:
        img = Image.open(path)
        target_mode = "RGB" if self.channels == 3 else "L"
        if img.mode != target_mode:
            img = img.convert(target_mode)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.records[idx]
        img = self._load(path)
        x = self.transform(img)
        return x, int(label), path


class PKBatchSampler(Sampler[List[int]]):
    """P×K class-balanced batch sampler.

    Each batch contains P identities × K samples per identity. Identities are
    sampled without replacement per epoch (when possible); if num_classes < P,
    we repeat with replacement. Within an identity, samples are drawn with
    replacement if fewer than K, otherwise without.

    Args:
        labels: [N] sequence of integer labels for each dataset record.
        P: number of identities per batch.
        K: number of samples per identity per batch.
        num_batches: explicit batches per epoch. If None, uses ceil(N / (P*K)).
        seed: RNG seed.
    """

    def __init__(
        self,
        labels: Sequence[int],
        P: int = 32,
        K: int = 4,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=None)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.P = int(P)
        self.K = int(K)
        self.seed = int(seed)
        self._epoch = 0

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.label_to_indices[int(lbl)].append(int(i))
        self.unique_labels = sorted(self.label_to_indices.keys())

        if num_batches is None:
            self.num_batches = max(1, int(np.ceil(len(self.labels) / (self.P * self.K))))
        else:
            self.num_batches = int(num_batches)

    @property
    def batch_size(self) -> int:
        return self.P * self.K

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.num_batches):
            if len(self.unique_labels) >= self.P:
                chosen = rng.choice(self.unique_labels, size=self.P, replace=False)
            else:
                chosen = rng.choice(self.unique_labels, size=self.P, replace=True)
            batch: List[int] = []
            for lbl in chosen:
                pool = self.label_to_indices[int(lbl)]
                if len(pool) >= self.K:
                    picks = rng.choice(pool, size=self.K, replace=False)
                else:
                    picks = rng.choice(pool, size=self.K, replace=True)
                batch.extend(int(p) for p in picks)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
