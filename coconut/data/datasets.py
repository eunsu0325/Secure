"""Dataset classes for COCONUT"""

import os
from typing import List, Optional, Union
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset

__all__ = [
    'BaseVeinDataset',
    'DualViewDataset',
    'SingleViewDataset',
    'MemoryDataset'
]


def _open_with_channels(path: str, channels: int) -> Image.Image:
    """Open image and convert to appropriate channels"""
    with Image.open(path) as img:
        if channels == 1:
            return img.convert('L')
        else:
            return img.convert('RGB')


class BaseVeinDataset(Dataset):
    """Base class for palm vein datasets"""

    def __init__(self,
                 paths: Union[List[str], str],
                 labels: Optional[List[int]] = None,
                 transform=None,
                 train: bool = True,
                 channels: int = 1,
                 dual_views: Optional[bool] = None):
        """
        Args:
            paths: List of image paths or path to txt file
            labels: List of labels (optional if paths is txt file)
            transform: Image transformations
            train: Training mode flag
            channels: Number of channels (1 for grayscale, 3 for RGB)
            dual_views: Whether to return dual views (defaults to train mode)
        """
        self.transform = transform
        self.train = train
        self.channels = channels
        self.dual_views = dual_views if dual_views is not None else train

        # Load paths and labels
        if isinstance(paths, str):
            # Load from txt file
            self.paths, self.labels = self._load_from_txt(paths)
        else:
            # Direct lists
            self.paths = paths
            if labels is not None:
                if torch.is_tensor(labels):
                    self.labels = labels.cpu().numpy()
                else:
                    self.labels = np.array(labels)
            else:
                raise ValueError("Labels must be provided if paths is a list")

        # Build class-to-indices mapping
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            label_int = int(label)
            if label_int not in self.class_to_indices:
                self.class_to_indices[label_int] = []
            self.class_to_indices[label_int].append(idx)

    def _load_from_txt(self, txt_path: str):
        """Load paths and labels from txt file"""
        paths = []
        labels = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                if len(item) >= 2:
                    paths.append(item[0])
                    labels.append(int(item[1]))

        return paths, np.array(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index])

        # Load image
        img = _open_with_channels(path, self.channels)

        # Apply transformations
        if self.dual_views:
            # Generate two different augmented views
            data1 = self.transform(img) if self.transform else img
            data2 = self.transform(img) if self.transform else img
            return [data1, data2], label
        else:
            # Single view
            data = self.transform(img) if self.transform else img
            return data, label


class DualViewDataset(BaseVeinDataset):
    """Dataset that always returns dual views for training"""

    def __init__(self, *args, **kwargs):
        kwargs['dual_views'] = True
        super().__init__(*args, **kwargs)


class SingleViewDataset(BaseVeinDataset):
    """Dataset that always returns single view for evaluation"""

    def __init__(self, *args, **kwargs):
        kwargs['dual_views'] = False
        kwargs['train'] = False
        super().__init__(*args, **kwargs)


class MemoryDataset(BaseVeinDataset):
    """Dataset for memory buffer samples (compatible with existing code)"""

    def __init__(self,
                 paths: List[str],
                 labels: List[int],
                 transform,
                 train: bool = True,
                 dual_views: Optional[bool] = None,
                 channels: int = 1):
        """
        Compatibility wrapper for existing MemoryDataset usage
        """
        super().__init__(
            paths=paths,
            labels=labels,
            transform=transform,
            train=train,
            channels=channels,
            dual_views=dual_views
        )