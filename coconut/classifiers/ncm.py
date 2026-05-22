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

    def __init__(self, normalize: bool = True):
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
        self.rejection_gate = 'top1_only'  # 'top1_only' | 'top1_margin'

        # S-norm (per-class Z-score on raw cosine)
        # cohort = unknown_dev features. cohort_mu/sigma는 _vectorize_means_dict와
        # 같은 (max_class+1,) 1D 텐서로 저장되어 forward에서 broadcast됨.
        self.snorm_enabled = False
        self.cohort_mu = None       # Tensor(C,)
        self.cohort_sigma = None    # Tensor(C,)
        self.snorm_min_sigma = 1e-2

    def state_dict(self):
        """register_buffer 외 plain 속성도 함께 저장합니다."""
        sd = super().state_dict()
        # S-norm
        sd['_custom_snorm_enabled'] = self.snorm_enabled
        sd['_custom_cohort_mu'] = self.cohort_mu
        sd['_custom_cohort_sigma'] = self.cohort_sigma
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        """체크포인트에서 상태를 로드합니다. _custom_ 키도 복원합니다."""
        # _custom_ 키 추출 (super()에 전달하면 unexpected key 에러)
        custom_keys = {k: state_dict.pop(k) for k in list(state_dict.keys()) if k.startswith('_custom_')}

        self.class_means = state_dict["class_means"]
        super().load_state_dict(state_dict, strict)
        if self.class_means is not None:
            for i in range(self.class_means.shape[0]):
                if (self.class_means[i] != 0).any():
                    self.class_means_dict[i] = self.class_means[i].clone()
        self.max_class = max(self.class_means_dict.keys()) if self.class_means_dict else -1

        # custom 상태 복원 (이전 체크포인트 하위 호환: 키 없으면 기본값 유지)
        # Mahalanobis 관련 _custom_score_mode / _custom_global_var 키는 이전
        # 체크포인트에는 있을 수 있으나 무시됨 (cosine-only)
        if custom_keys:
            self.snorm_enabled = custom_keys.get('_custom_snorm_enabled', False)
            self.cohort_mu = custom_keys.get('_custom_cohort_mu', None)
            self.cohort_sigma = custom_keys.get('_custom_cohort_sigma', None)

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
    def forward(self, x, apply_snorm: bool = True):
        """
         최적화된 NCM 분류

        normalize=True: 코사인 유사도 (정규화 후 내적)
        normalize=False: 유클리디안 거리 (제곱 거리 사용)

        apply_snorm: True면 S-norm 적용 (snorm_enabled일 때).
                     진단/시각화용은 False로 raw cosine 유지.
        """
        # NCM이 비어있으면 빈 점수 반환
        if self.class_means_dict == {}:
            return torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)

        # dtype 일치 보장 (fp16/AMP 지원)
        M = self.class_means.to(device=x.device, dtype=x.dtype)

        if self.normalize:
            # 코사인 유사도 기반
            x = F.normalize(x, p=2, dim=1, eps=1e-12)
            scores = x @ M.T  # (B, C) raw cosine
            # S-norm: per-class Z-score 정규화
            if apply_snorm and self.snorm_enabled and self.cohort_mu is not None:
                mu = self.cohort_mu.to(device=scores.device, dtype=scores.dtype)
                sigma = self.cohort_sigma.to(device=scores.device, dtype=scores.dtype)
                C_scores = scores.shape[1]
                C_cohort = mu.shape[0]
                if C_cohort < C_scores:
                    # 새 클래스 추가됨 — cohort 미계산 클래스는 identity (mu=0, σ=1)
                    pad = C_scores - C_cohort
                    mu = torch.cat([mu, torch.zeros(pad, device=mu.device, dtype=mu.dtype)])
                    sigma = torch.cat([sigma, torch.ones(pad, device=sigma.device, dtype=sigma.dtype)])
                scores = (scores - mu.unsqueeze(0)) / sigma.unsqueeze(0)
            return scores
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

        # L2 정규화: cosine 모드는 항상 적용
        if self.normalize:
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

    def set_cohort_stats(self, cohort_mu_dict: Dict[int, float], cohort_sigma_dict: Dict[int, float]):
        """S-norm용 per-class cohort 평균/표준편차 설정.
        class_means와 같은 (max_class+1,) 텐서로 vectorize."""
        if self.class_means is None:
            return
        device = self.class_means.device
        dtype = self.class_means.dtype
        C = self.class_means.shape[0]
        mu = torch.zeros(C, device=device, dtype=dtype)
        sigma = torch.ones(C, device=device, dtype=dtype)
        for k, v in cohort_mu_dict.items():
            if 0 <= k < C:
                mu[k] = float(v)
        for k, v in cohort_sigma_dict.items():
            if 0 <= k < C:
                sigma[k] = max(float(v), self.snorm_min_sigma)
        self.cohort_mu = mu
        self.cohort_sigma = sigma
        self.snorm_enabled = True
        # 진단: cohort 통계 확인용
        nz = (sigma != 1.0).sum().item()
        print(f"   [S-norm] cohort set: "
              f"C={C}, nz_classes={nz}, "
              f"mu[mean={mu.mean().item():.3e}, std={mu.std().item():.3e}], "
              f"sigma[mean={sigma.mean().item():.3e}, min={sigma.min().item():.3e}]")

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

        scores = self.forward(x, apply_snorm=False)  # (B, C) raw cosine — margin은 raw 기준
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

    @torch.no_grad()
    def predict_openset(self, x):
        """오픈셋 예측 (모드별 분기: top1_margin / top1_only)"""
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)

        # --- top1_margin 이중 게이트 ---
        if self.rejection_gate == 'top1_margin':
            result = self.compute_dual_gate_scores(x)
            pred = result['pred_ids']
            accept = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

            if self.tau_cos is not None:
                accept &= result['cosine_max'] >= self.tau_cos
            if self.tau_margin is not None:
                accept &= result['margin'] >= self.tau_margin

            pred[~accept] = self.unknown_id
            return pred

        # --- top1_only 모드 ---
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
