"""Tongji ROI dataset for exp1_baselines.

Image transform pipeline:
  1. Load BMP (grayscale or RGB)
  2. Convert to 3-channel RGB if needed
  3. Resize to 112x112
  4. ToTensor (uint8 → float in [0,1])
  5. Normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] → final range [-1, 1]
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


def build_baseline_transform(image_size: int = 112) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class TongjiROIDataset(Dataset):
    """Generic Tongji ROI dataset reading (image_path, label) pairs.

    Args:
        records: list of (absolute_path, label) tuples.
        transform: torchvision transform applied to each loaded image.
    """

    def __init__(
        self,
        records: Sequence[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 112,
    ) -> None:
        self.records: List[Tuple[str, int]] = [
            (str(p), int(l)) for p, l in records
        ]
        self.transform = transform if transform is not None else build_baseline_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.records[idx]
        img = self._load_rgb(path)
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
