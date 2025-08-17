# scr/ncm_classifier.py
import numpy as np  # ⭐️ 에너지 스코어를 위해 추가
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NCMClassifier(nn.Module):
    """
    NCM (Nearest Class Mean) Classifier - 최적화 버전.
    
    🚀 최적화: cdist 대신 행렬곱 사용 (2-3배 빠름)
    ✅ normalize=True: 코사인 유사도 기반
    ⭐️ 에너지 스코어 기반 오픈셋 지원
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.register_buffer("class_means", None)
        self.class_means_dict = {}
        self.normalize = normalize
        self.max_class = -1
        
        # 오픈셋 관련
        self.tau_s = None            # 전역 임계치
        # ⚡️ self.use_margin = False  # 제거: 에너지 스코어로 대체
        # ⚡️ self.tau_m = 0.05        # 제거: 에너지 스코어로 대체
        self.unknown_id = -1        # Unknown 클래스 ID
        
        # ⚡️ Z-norm 관련 완전 제거 (복잡성 대비 효과 미미)
        # self.use_znorm = False
        # self.impostor_stats = {}
        
        # ⭐️ 에너지 스코어 설정 추가
        self.use_energy = False      # 기본값 False (기존 코드 호환성)
        self.energy_T = 0.15         # Temperature (SupCon 0.07보다 높게)
        self.energy_k_mode = 'sqrt'  # 'sqrt', 'fixed', 'log'
        self.energy_k_fixed = 10     # k_mode='fixed'일 때 사용

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

    # ⚡️ forward_cdist 제거 (테스트용이었으나 더 이상 불필요)
    # ⚡️ 90-105 라인 전체 제거

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

    # ⚡️ init_missing_classes 제거 (사용하지 않음)
    # ⚡️ adaptation 제거 (Avalanche 전용)
    
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
    
    # ⭐️ === 에너지 스코어 메서드 추가 ===
    
    @torch.no_grad()
    def compute_energy_score(self, x, k=None, T=None):
        """
        ⭐️ 에너지 기반 게이트 스코어 계산
        
        Args:
            x: (B, D) 특징 벡터
            k: Top-k 파라미터 (None이면 자동 결정)
            T: Temperature (None이면 self.energy_T 사용)
            
        Returns:
            (B,) 게이트 스코어 (높을수록 등록자스러움)
        """
        # NCM이 비어있으면 낮은 점수 반환
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1000.0, device=x.device, dtype=x.dtype)
        
        # 코사인 유사도 계산
        scores = self.forward(x)  # (B, C), [-1, 1]
        B, C = scores.shape
        
        # k 결정
        if k is None:
            if self.energy_k_mode == 'sqrt':
                k = max(3, min(20, int(np.sqrt(C))))
            elif self.energy_k_mode == 'log':
                k = max(3, min(20, int(np.log(C) * 3)))
            else:  # fixed
                k = self.energy_k_fixed
        
        # Temperature 설정
        T = T if T is not None else self.energy_T
        
        # k를 클래스 수로 제한
        k_prime = min(k, C)
        
        # Top-k 추출
        topk_values, _ = scores.topk(k_prime, dim=1)  # (B, k')
        
        # 수치 안정 LogSumExp
        z = topk_values / T  # (B, k')
        max_z = z.max(dim=1, keepdim=True).values  # (B, 1)
        
        # energy = T * log(sum(exp(s/T)))
        energy = T * (max_z.squeeze(1) + torch.log(torch.exp(z - max_z).sum(dim=1)))
        
        # k 보정 (정규화)
        gate_score = energy - T * np.log(k_prime)
        
        return gate_score
    
    @torch.no_grad()
    def compute_energy_masked(self, x, labels, k=None, T=None):
        """
        ⭐️ 자기 클래스를 제외한 Top-k 에너지 계산 (Between impostor용)
        
        Args:
            x: (B, D) 특징 벡터
            labels: (B,) 정답 레이블
            
        Returns:
            (B,) 마스킹된 게이트 스코어
        """
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1000.0, device=x.device, dtype=x.dtype)
        
        scores = self.forward(x)  # (B, C)
        B, C = scores.shape
        
        # k 결정
        if k is None:
            if self.energy_k_mode == 'sqrt':
                k = max(3, min(20, int(np.sqrt(C))))
            else:
                k = self.energy_k_fixed
        
        T = T if T is not None else self.energy_T
        
        # 자기 클래스 제외하므로 C-1이 최대
        k_prime = max(1, min(k, C - 1))
        
        # 자기 클래스 마스킹
        mask = F.one_hot(labels, num_classes=C).bool()  # (B, C)
        scores_masked = scores.masked_fill(mask, float('-inf'))  # (B, C)
        
        # Top-k 추출 (자기 제외)
        topk_values, _ = scores_masked.topk(k_prime, dim=1)  # (B, k')
        
        # 수치 안정 LogSumExp
        z = topk_values / T
        max_z = z.max(dim=1, keepdim=True).values
        energy = T * (max_z.squeeze(1) + torch.log(torch.exp(z - max_z).sum(dim=1)))
        
        # k 보정
        gate_score = energy - T * np.log(k_prime)
        
        return gate_score
    
    # === 오픈셋 관련 메서드 (수정) ===
    
    def set_thresholds(self, tau_s: float):
        """
        오픈셋 임계치 설정 (심플화)
        ⚡️ 마진 파라미터 제거
        """
        self.tau_s = float(tau_s)
        # ⚡️ use_margin, tau_m 관련 제거
    
    # ⚡️ enable_znorm 제거 (Z-norm 미사용)
    # ⚡️ update_impostor_stats 제거 (Z-norm 미사용)
    
    def set_energy_config(self, use_energy=True, T=0.15, k_mode='sqrt', k_fixed=10):
        """
        ⭐️ 에너지 스코어 설정
        """
        self.use_energy = use_energy
        self.energy_T = T
        self.energy_k_mode = k_mode
        self.energy_k_fixed = k_fixed
        
        if use_energy:
            print(f"⚡ Energy mode enabled: T={T}, k_mode={k_mode}")
    
    @torch.no_grad()
    def predict_openset(self, x):
        """
        오픈셋 예측 (에너지 또는 최댓값 모드)
        """
        # NCM이 비어있으면 모두 -1 반환
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        
        # ⭐️ 에너지 모드
        if self.use_energy:
            return self.predict_openset_energy(x)
        
        # 기존 최댓값 모드 (Z-norm, 마진 제거)
        scores = self.forward(x)  # (B, C)
        
        # Top-1 추출
        top1 = scores.topk(1, dim=1)
        max_score = top1.values[:, 0]
        pred = top1.indices[:, 0]
        
        # ⚡️ Z-norm 로직 제거 (241-251 라인)
        # ⚡️ 마진 로직 제거 (261-264 라인)
        
        # 임계치 적용
        if self.tau_s is not None:
            accept = max_score >= self.tau_s
        else:
            accept = torch.ones_like(max_score, dtype=torch.bool)
        
        pred[~accept] = self.unknown_id
        
        return pred
    
    @torch.no_grad()
    def predict_openset_energy(self, x):
        """
        ⭐️ 에너지 스코어 기반 오픈셋 예측
        """
        # NCM이 비어있으면 모두 거부
        if len(self.class_means_dict) == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        
        # 게이트 스코어 계산
        gate_scores = self.compute_energy_score(x)
        
        # 임계치 비교
        if self.tau_s is not None:
            accept = gate_scores >= self.tau_s
        else:
            accept = torch.ones(gate_scores.shape[0], dtype=torch.bool, device=x.device)
        
        # 클래스 예측 (argmax는 그대로)
        scores = self.forward(x)
        pred_classes = scores.argmax(dim=1)
        
        # 거부된 샘플은 -1
        pred_classes[~accept] = self.unknown_id
        
        return pred_classes
    
    @torch.no_grad()
    def get_openset_scores(self, x):
        """
        오픈셋 점수 상세 정보 반환 (디버깅용)
        ⭐️ 에너지 스코어 정보 추가
        """
        # NCM이 비어있으면 빈 결과 반환
        if len(self.class_means_dict) == 0:
            return {
                'scores': torch.zeros((x.shape[0], 0), device=x.device),
                'top_scores': torch.zeros(x.shape[0], device=x.device),
                'gate_scores': torch.full((x.shape[0],), -1000.0, device=x.device) if self.use_energy else None,  # ⭐️
                'predictions': torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device),
                'accept_mask': torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
                'tau_s': self.tau_s,
                'mode': 'energy' if self.use_energy else 'max'  # ⭐️
            }
        
        scores = self.forward(x)
        top1 = scores.topk(1, dim=1)
        
        # ⭐️ 에너지 스코어 추가
        gate_scores = None
        if self.use_energy:
            gate_scores = self.compute_energy_score(x)
        
        pred = self.predict_openset(x)
        accept_mask = pred != self.unknown_id
        
        return {
            'scores': scores,
            'top_scores': top1.values[:, 0],
            'gate_scores': gate_scores,  # ⭐️
            'predictions': pred,
            'accept_mask': accept_mask,
            'tau_s': self.tau_s,
            'mode': 'energy' if self.use_energy else 'max'  # ⭐️
        }
    
    # ⚡️ verify_equivalence 제거 (테스트용이었으나 불필요)


# 테스트 코드
if __name__ == "__main__":
    import time
    
    print("=== 코사인 NCM 테스트 ===")
    ncm_cos = NCMClassifier(normalize=True)
    
    # 빈 NCM 테스트
    print("\n빈 NCM 테스트:")
    x_test = torch.randn(10, 128)
    pred_empty = ncm_cos.predict(x_test)
    print(f"빈 NCM 예측: {pred_empty} (모두 -1이어야 함)")
    
    # 정규화된 프로토타입 생성
    class_means = {}
    for i in range(10):
        class_means[i] = F.normalize(torch.randn(128), p=2, dim=0)
    ncm_cos.replace_class_means_dict(class_means)
    
    x = torch.randn(100, 128)
    
    # ⭐️ === 에너지 스코어 테스트 ===
    print("\n=== 에너지 스코어 테스트 ===")
    
    # 에너지 모드 활성화
    ncm_cos.set_energy_config(use_energy=True, T=0.15, k_mode='sqrt')
    ncm_cos.set_thresholds(tau_s=0.0)
    
    # 게이트 스코어 계산
    gate_scores = ncm_cos.compute_energy_score(x)
    print(f"Gate scores: min={gate_scores.min():.3f}, max={gate_scores.max():.3f}, mean={gate_scores.mean():.3f}")
    
    # 모드 비교
    ncm_cos.use_energy = False
    pred_max = ncm_cos.predict_openset(x)
    
    ncm_cos.use_energy = True
    pred_energy = ncm_cos.predict_openset(x)
    
    print(f"Max mode rejections: {(pred_max == -1).sum()}/{len(x)}")
    print(f"Energy mode rejections: {(pred_energy == -1).sum()}/{len(x)}")
    
    # 자기 클래스 마스킹 테스트
    labels = torch.randint(0, 10, (20,))
    x_test = torch.randn(20, 128)
    masked_scores = ncm_cos.compute_energy_masked(x_test, labels)
    print(f"Masked gate scores: mean={masked_scores.mean():.3f}")
    
    # 상세 정보
    details = ncm_cos.get_openset_scores(x)
    print(f"Mode: {details['mode']}, Tau_s: {details['tau_s']}")