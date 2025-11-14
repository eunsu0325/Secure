# scr/ncm_classifier.py
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NCMClassifier(nn.Module):
    """
    NCM (Nearest Class Mean) Classifier - 최적화 버전.
    
    🚀 최적화: cdist 대신 행렬곱 사용 (2-3배 빠름)
    ✅ normalize=True: 코사인 유사도 기반
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.register_buffer("class_means", None)
        self.class_means_dict = {}
        self.normalize = normalize
        self.max_class = -1
        
        # 오픈셋 관련
        self.tau_s = None            # 전역 임계치
        self.unknown_id = -1         # Unknown 클래스 ID

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
        🚀 최적화된 NCM 분류
        
        normalize=True: 코사인 유사도 (정규화 후 내적)
        normalize=False: 유클리디안 거리 (제곱 거리 사용)
        """
        # NCM이 비어있으면 빈 점수 반환
        if self.class_means_dict == {}:
            return torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)
        
        # dtype 일치 보장 (fp16/AMP 지원)
        M = self.class_means.to(device=x.device, dtype=x.dtype)
        
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

    def update_class_means_dict(self, class_means_dict: Dict[int, Tensor], momentum: float = 0.5):
        """
        클래스 평균을 업데이트합니다. (EMA 지원)
        나중에 EMA 실험을 위해 유지
        """
        assert 0 <= momentum <= 1
        assert isinstance(class_means_dict, dict)
        
        for k, v in class_means_dict.items():
            if k not in self.class_means_dict or (self.class_means_dict[k] == 0).all():
                # 새로운 클래스
                self.class_means_dict[k] = class_means_dict[k].clone()
            else:
                # 기존 클래스 업데이트
                device = self.class_means_dict[k].device
                self.class_means_dict[k] = (
                    momentum * class_means_dict[k].to(device)
                    + (1 - momentum) * self.class_means_dict[k]
                )
        
        # 방어적 정규화 (normalize=True일 때)
        if self.normalize:
            for k in self.class_means_dict:
                self.class_means_dict[k] = F.normalize(
                    self.class_means_dict[k], p=2, dim=0, eps=1e-12
                )
        
        self._vectorize_means_dict()

    def replace_class_means_dict(self, class_means_dict: Dict[int, Tensor]):
        """
        기존 평균을 완전히 교체합니다.
        현재 주로 사용되는 메서드
        """
        assert isinstance(class_means_dict, dict)
        
        self.class_means_dict = {k: v.clone() for k, v in class_means_dict.items()}
        
        # 방어적 정규화 (normalize=True일 때)
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
    
    def set_thresholds(self, tau_s: float):
        """오픈셋 임계치 설정"""
        self.tau_s = float(tau_s)
    
    @torch.no_grad()
    def predict_openset(self, x):
        """오픈셋 예측 (최댓값 모드)"""
        # NCM이 비어있으면 모두 -1 반환
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        
        # 기존 최댓값 모드
        scores = self.forward(x)  # (B, C)
        
        # Top-1 추출
        top1 = scores.topk(1, dim=1)
        max_score = top1.values[:, 0]
        pred = top1.indices[:, 0]
        
        # 임계치 적용
        if self.tau_s is not None:
            accept = max_score >= self.tau_s
        else:
            accept = torch.ones_like(max_score, dtype=torch.bool)
        
        pred[~accept] = self.unknown_id
        
        return pred
    
    @torch.no_grad()
    def get_openset_scores(self, x):
        """오픈셋 점수 상세 정보 반환 (디버깅용)"""
        # NCM이 비어있으면 빈 결과 반환
        if len(self.class_means_dict) == 0:
            return {
                'scores': torch.zeros((x.shape[0], 0), device=x.device),
                'top_scores': torch.zeros(x.shape[0], device=x.device),
                'predictions': torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device),
                'accept_mask': torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
                'tau_s': self.tau_s,
                'mode': 'max'
            }
        
        scores = self.forward(x)
        top1 = scores.topk(1, dim=1)
        
        pred = self.predict_openset(x)
        accept_mask = pred != self.unknown_id
        
        return {
            'scores': scores,
            'top_scores': top1.values[:, 0],
            'predictions': pred,
            'accept_mask': accept_mask,
            'tau_s': self.tau_s,
            'mode': 'max'
        }