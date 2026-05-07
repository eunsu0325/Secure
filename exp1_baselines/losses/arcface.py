"""ArcFace classification head + loss.

Attribution:
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition" (CVPR 2019).
    Implementation reference: deepinsight/insightface/recognition/arcface_torch/losses.py
    and partial_fc.py.

This head is used during ArcFace backbone training only and discarded after training.

Numerical stability:
  - cosine logits are clamped to [-1+eps, 1-eps] before arccos to avoid NaN
  - When AMP is used, ArcFace margin computation runs in FP32
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """ArcFace classification head.

    Args:
        embedding_dim: feature dim from backbone (default 512).
        num_classes: number of training identities.
        margin: angular margin m (default 0.5 rad).
        scale: feature scale s (default 48).
        eps: clamp epsilon for arccos numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 320,
        margin: float = 0.5,
        scale: float = 48.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = float(margin)
        self.scale = float(scale)
        self.eps = float(eps)

        # Classifier weight W: [num_classes, embedding_dim]
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace scaled logits.

        Args:
            features: [B, D] backbone features (NOT yet L2-normalized; we normalize here).
            labels:   [B] long tensor of class indices.

        Returns:
            scaled_logits: [B, num_classes] — feed into nn.CrossEntropyLoss.
        """
        # Compute in FP32 for AMP safety.
        features_fp32 = features.float()
        weight_fp32 = self.weight.float()

        feat_norm = F.normalize(features_fp32, p=2, dim=1)
        weight_norm = F.normalize(weight_fp32, p=2, dim=1)

        # Cosine logits: [B, num_classes]
        cosine = F.linear(feat_norm, weight_norm)
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # Apply additive angular margin to ground-truth class only.
        theta = torch.acos(cosine)
        target_mask = F.one_hot(labels, num_classes=self.num_classes).bool()
        theta_margin = torch.where(target_mask, theta + self.margin, theta)
        cosine_margin = torch.cos(theta_margin)

        logits = cosine_margin * self.scale
        return logits
