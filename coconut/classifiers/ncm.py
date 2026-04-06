"""
NCM (Nearest Class Mean) Classifier for COCONUT
최적화된 거리 기반 분류기
"""

from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NCMClassifier(nn.Module):
    """
    NCM (Nearest Class Mean) Classifier - 최적화 버전.

     최적화: cdist 대신 행렬곱 사용 (2-3배 빠름)
     normalize=True: 코사인 유사도 기반
    """

    def __init__(self, normalize: bool = True, score_mode: str = 'cosine', var_reg_alpha: float = 1e-4):
        super().__init__()
        self.register_buffer("class_means", None)
        self.class_means_dict = {}
        self.normalize = normalize
        self.max_class = -1

        # 오픈셋 관련
        self.tau_s = None            # 전역 임계치 (레거시 호환)
        self.unknown_id = -1         # Unknown 클래스 ID

        # 이중 게이트 (cosine + margin)
        self.tau_cos = None          # cosine threshold
        self.tau_margin = None       # margin threshold (top1 - top2)
        self.rejection_gate = 'cosine_only'  # 'cosine_only' | 'cosine_margin'

        # Mahalanobis 관련
        self.score_mode = score_mode          # 'cosine' | 'mahalanobis'
        self.var_reg_alpha = var_reg_alpha
        self.global_var = None                # Tensor(D,) — shared diagonal variance

        # GHOST 관련 (z-score 기반 rejection, 레거시)
        self.ghost_enabled = False
        self.ghost_class_means_raw = {}       # {class_id: Tensor(D,)} raw feature mean
        self.ghost_class_stds_raw = {}        # {class_id: Tensor(D,)} raw feature std
        self.ghost_global_std_raw = None      # Tensor(D,) shrinkage target
        self.ghost_class_counts = {}          # {class_id: int} augmented sample count
        self.ghost_shrinkage_min_n = 10       # shrinkage 기준 샘플 수

    def load_state_dict(self, state_dict, strict: bool = True):
        """체크포인트에서 상태를 로드합니다."""
        self.class_means = state_dict["class_means"]
        super().load_state_dict(state_dict, strict)
        if self.class_means is not None:
            for i in range(self.class_means.shape[0]):
                if (self.class_means[i] != 0).any():
                    self.class_means_dict[i] = self.class_means[i].clone()
        self.max_class = max(self.class_means_dict.keys()) if self.class_means_dict else -1

    def _vectorize_means_dict(self):
        """딕셔너리를 텐서로 변환합니다."""
        if self.class_means_dict == {}:
            return

        max_class = max(self.class_means_dict.keys())
        self.max_class = max(max_class, self.max_class)

        first_mean = list(self.class_means_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device

        self.class_means = torch.zeros(self.max_class + 1, feature_size).to(device)

        for k, v in self.class_means_dict.items():
            self.class_means[k] = self.class_means_dict[k].clone()

    @torch.no_grad()
    def forward(self, x):
        """
         최적화된 NCM 분류

        score_mode='mahalanobis': whitened euclidean (shared diagonal Mahalanobis)
        normalize=True: 코사인 유사도 (정규화 후 내적)
        normalize=False: 유클리디안 거리 (제곱 거리 사용)
        """
        # NCM이 비어있으면 빈 점수 반환
        if self.class_means_dict == {}:
            return torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)

        # dtype 일치 보장 (fp16/AMP 지원)
        M = self.class_means.to(device=x.device, dtype=x.dtype)

        if self.score_mode == 'mahalanobis' and self.global_var is not None:
            # L2 normalize → whitening → negative squared euclidean
            # L2 norm: magnitude 차이 제거 (cosine의 장점 유지)
            # whitening: 차원별 중요도 가중 (Mahalanobis의 장점 추가)
            x = F.normalize(x, p=2, dim=1, eps=1e-12)

            inv_std = 1.0 / torch.sqrt(
                self.global_var.to(device=x.device, dtype=x.dtype) + self.var_reg_alpha
            )  # (D,)
            x_w = x * inv_std                                # (B, D)
            M_w = M * inv_std                                # (C, D)
            x2 = (x_w * x_w).sum(dim=1, keepdim=True)       # (B, 1)
            m2 = (M_w * M_w).sum(dim=1, keepdim=False)      # (C,)
            xm = x_w @ M_w.T                                # (B, C)
            scores = -(x2 + m2.unsqueeze(0) - 2 * xm)      # (B, C)
            return scores  # 높을수록 가까움 (negative Mahalanobis distance)

        if self.normalize:
            # 코사인 유사도 기반
            x = F.normalize(x, p=2, dim=1, eps=1e-12)
            scores = x @ M.T  # (B, C)
            return scores  # 높을수록 가까움
        else:
            # 유클리디안 거리 기반
            x2 = (x * x).sum(dim=1, keepdim=True)      # (B, 1)
            m2 = (M * M).sum(dim=1, keepdim=False)      # (C,)
            xm = x @ M.T                                # (B, C)
            scores = -(x2 + m2.unsqueeze(0) - 2 * xm)  # (B, C)
            return scores  # 높을수록 가까움 (negative distance)

    def replace_class_means_dict(self, class_means_dict: Dict[int, Tensor]):
        """
        기존 평균을 완전히 교체합니다.
        현재 주로 사용되는 메서드
        """
        assert isinstance(class_means_dict, dict)

        self.class_means_dict = {k: v.clone() for k, v in class_means_dict.items()}

        # L2 정규화: cosine 모드는 항상, mahalanobis 모드도 동일하게 적용
        # (forward()에서 probe를 L2 normalize하므로 mean도 맞춰야 함)
        if self.normalize or self.score_mode == 'mahalanobis':
            for k in self.class_means_dict:
                self.class_means_dict[k] = F.normalize(
                    self.class_means_dict[k], p=2, dim=0, eps=1e-12
                )

        self._vectorize_means_dict()

    def predict(self, x):
        """클래스 예측을 반환합니다."""
        # NCM이 비어있으면 -1 반환
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)

        scores = self.forward(x)
        return scores.argmax(dim=1)

    def get_num_classes(self):
        """현재 저장된 클래스 수를 반환합니다."""
        return len(self.class_means_dict)

    def get_class_means(self):
        """현재 저장된 클래스 평균들을 반환합니다."""
        return self.class_means_dict.copy()

    def set_thresholds(self, tau_s: float = None, tau_cos: float = None, tau_margin: float = None):
        """오픈셋 임계치 설정"""
        if tau_s is not None:
            self.tau_s = float(tau_s)
        if tau_cos is not None:
            self.tau_cos = float(tau_cos)
        if tau_margin is not None:
            self.tau_margin = float(tau_margin)

    def set_global_var(self, global_var: Tensor):
        """Global shared diagonal variance 설정 (Mahalanobis용)"""
        self.global_var = global_var.clone()

    def set_ghost_stats(self, class_means_raw, class_stds_raw, global_std_raw, class_counts):
        """GHOST용 per-class raw feature 통계 설정"""
        self.ghost_class_means_raw = {k: v.clone() for k, v in class_means_raw.items()}
        self.ghost_class_stds_raw = {k: v.clone() for k, v in class_stds_raw.items()}
        self.ghost_global_std_raw = global_std_raw.clone() if global_std_raw is not None else None
        self.ghost_class_counts = dict(class_counts)

    @torch.no_grad()
    def compute_dual_gate_scores(self, x):
        """
        이중 게이트용 스코어 계산.
        Returns: dict with 'cosine_max', 'margin', 'pred_ids'
        """
        if len(self.class_means_dict) == 0:
            B = x.shape[0]
            return {
                'cosine_max': torch.zeros(B, device=x.device),
                'margin': torch.zeros(B, device=x.device),
                'pred_ids': torch.full((B,), -1, dtype=torch.long, device=x.device),
            }

        scores = self.forward(x)  # (B, C) cosine scores
        if scores.shape[1] >= 2:
            topk = scores.topk(2, dim=1)
            cosine_max = topk.values[:, 0]
            cosine_2nd = topk.values[:, 1]
            margin = cosine_max - cosine_2nd
            pred_ids = topk.indices[:, 0]
        else:
            cosine_max = scores[:, 0]
            margin = cosine_max
            pred_ids = torch.zeros(scores.shape[0], dtype=torch.long, device=x.device)

        return {
            'cosine_max': cosine_max,
            'margin': margin,
            'pred_ids': pred_ids,
        }

    def _compute_ghost_scores(self, x_raw, pred_ids, cosine_scores):
        """
        GHOST score 계산: γ = z_k̂ / s  (AAAI 2025, Eq. 3-4)

        Args:
            x_raw: (B, D) raw features (L2 norm 전)
            pred_ids: (B,) predicted class indices
            cosine_scores: (B,) z_k̂ (cosine max score)
        Returns:
            (B,) GHOST scores
        """
        ghost_scores = torch.zeros(x_raw.shape[0], device=x_raw.device, dtype=x_raw.dtype)

        for i in range(x_raw.shape[0]):
            k = pred_ids[i].item()

            if k not in self.ghost_class_stds_raw:
                ghost_scores[i] = cosine_scores[i]
                continue

            mu_k = self.ghost_class_means_raw[k].to(device=x_raw.device, dtype=x_raw.dtype)
            std_k = self.ghost_class_stds_raw[k].to(device=x_raw.device, dtype=x_raw.dtype)

            # Shrinkage: 샘플 적으면 global std와 혼합
            n_k = self.ghost_class_counts.get(k, 1)
            if self.ghost_global_std_raw is not None:
                lam = min(n_k / self.ghost_shrinkage_min_n, 1.0)
                global_std = self.ghost_global_std_raw.to(device=x_raw.device, dtype=x_raw.dtype)
                std_k = lam * std_k + (1 - lam) * global_std

            # Eq. 3: s = mean_d |φ_d - μ_{k̂,d}| / σ_{k̂,d}  (차원 정규화)
            s = (torch.abs(x_raw[i] - mu_k) / (std_k + 1e-12)).mean()

            # Eq. 4: γ = z_k̂ / s
            ghost_scores[i] = cosine_scores[i] / (s + 1e-12)

        return ghost_scores

    @torch.no_grad()
    def compute_ghost_max_scores(self, x_raw):
        """
        캘리브레이션/평가용: GHOST score 배열 반환.
        x_raw: (B, D) raw features
        """
        if len(self.class_means_dict) == 0:
            return torch.zeros(x_raw.shape[0], device=x_raw.device)

        scores = self.forward(x_raw)  # cosine (내부 L2 norm)
        top1 = scores.topk(1, dim=1)
        max_score = top1.values[:, 0]
        pred = top1.indices[:, 0]

        if self.ghost_enabled and self.ghost_class_stds_raw:
            return self._compute_ghost_scores(x_raw, pred, max_score)
        return max_score

    @torch.no_grad()
    def predict_openset(self, x):
        """오픈셋 예측 (모드별 분기: cosine_margin / GHOST 레거시 / cosine_only)"""
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)

        # --- cosine_margin 이중 게이트 ---
        if self.rejection_gate == 'cosine_margin':
            result = self.compute_dual_gate_scores(x)
            pred = result['pred_ids']
            accept = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

            if self.tau_cos is not None:
                accept &= result['cosine_max'] >= self.tau_cos
            if self.tau_margin is not None:
                accept &= result['margin'] >= self.tau_margin

            pred[~accept] = self.unknown_id
            return pred

        # --- 기존 GHOST 모드 (레거시) ---
        if self.ghost_enabled and self.ghost_class_stds_raw:
            scores = self.forward(x)
            top1 = scores.topk(1, dim=1)
            max_score = top1.values[:, 0]
            pred = top1.indices[:, 0]
            ghost_scores = self._compute_ghost_scores(x, pred, max_score)
            if self.tau_s is not None:
                accept = ghost_scores >= self.tau_s
            else:
                accept = torch.ones_like(ghost_scores, dtype=torch.bool)
            pred[~accept] = self.unknown_id
            return pred

        # --- cosine_only 모드 ---
        scores = self.forward(x)
        top1 = scores.topk(1, dim=1)
        max_score = top1.values[:, 0]
        pred = top1.indices[:, 0]
        tau = self.tau_cos if self.tau_cos is not None else self.tau_s
        if tau is not None:
            accept = max_score >= tau
            pred[~accept] = self.unknown_id
        return pred

    @torch.no_grad()
    def similarity(self, x, class_id: int):
        """
        Calculate similarity between input and specific class.

        Args:
            x: Input features (B, D) where B is batch size, D is feature dimension
            class_id: Target class ID to compute similarity with

        Returns:
            Similarity score (float) or None if class not found
        """
        # Check if class exists
        if class_id not in self.class_means_dict:
            return None

        # Get all scores
        scores = self.forward(x)  # (B, num_classes)

        # Return similarity for specific class
        if scores.shape[0] == 1:
            return float(scores[0, class_id])
        else:
            # If batch size > 1, return mean similarity
            return float(scores[:, class_id].mean())
