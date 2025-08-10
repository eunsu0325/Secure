# 🐋 scr/threshold_calculator.py (신규)
"""
EER 기반 임계치 계산 및 스무딩
- 코사인 유사도 기준
- EMA 스무딩 + 변화폭 제한
- 선택적 마진 자동 조정
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Optional, Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)  # interp1d 경고 무시


class ThresholdCalibrator:
    """
    🐋 전역 임계치(τ_s) 캘리브레이터
    - 코사인 유사도 기반(기본)
    - EER 지점 임계치 계산 + EMA 스무딩 + 변화폭 제한
    - (옵션) 마진 자동 조정
    """
    
    def __init__(
        self,
        mode: str = "cosine",                    # "cosine" 또는 "euclidean"
        alpha: float = 0.2,                      # EMA 계수
        max_delta: float = 0.03,                 # 한 번에 바뀌는 최대 변화폭
        clip_range: Optional[Tuple[float, float]] = (-1.0, 1.0),  # 코사인 기본 범위
        use_auto_margin: bool = False,           # 마진 자동 조정 여부
        margin_init: float = 0.05,               # 초기 마진값
        margin_bounds: Tuple[float, float] = (0.02, 0.10),  # 마진 범위
        margin_step_up: float = 0.01,            # FAR 높을 때 증가폭
        margin_step_down: float = 0.005,         # FAR 낮을 때 감소폭
        far_target: Optional[float] = None,      # auto-margin이 쓸 목표 FAR
        min_samples: int = 10,                   # 최소 샘플 수
    ):
        self.mode = mode
        self.alpha = alpha
        self.max_delta = max_delta
        self.clip_range = clip_range
        
        # 🐋 마진 관련
        self.use_auto_margin = use_auto_margin
        self.tau_m = margin_init
        self.margin_bounds = margin_bounds
        self.margin_step_up = margin_step_up
        self.margin_step_down = margin_step_down
        self.far_target = far_target
        
        # 🐋 최소 샘플 요구사항
        self.min_samples = min_samples
        
        # 🐋 히스토리 추적
        self.history: List[Dict] = []
        self.tau_s_current: Optional[float] = None
        
    def compute_eer_threshold(self, genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
        """
        🐋 EER 임계치 계산
        
        Args:
            genuine: genuine 페어의 유사도 점수 (높을수록 좋음)
            impostor: impostor 페어의 유사도 점수 (낮을수록 좋음)
            
        Returns:
            (tau_new, eer): 새로운 임계치와 EER
        """
        # 🐋 최소 샘플 체크
        if len(genuine) < self.min_samples or len(impostor) < self.min_samples:
            print(f"⚠️ Not enough samples: genuine={len(genuine)}, impostor={len(impostor)}")
            return self.tau_s_current if self.tau_s_current else 0.7, 0.0
        
        # 🐋 레이블 생성: genuine=1, impostor=0
        y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
        s = np.concatenate([genuine, impostor])
        
        # 🐋 ROC curve 계산
        fpr, tpr, thr = roc_curve(y, s)
        fnr = 1 - tpr
        
        # 🐋 EER 지점 찾기
        try:
            # 정확한 EER 지점 보간
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            tau = float(interp1d(fpr, thr)(eer))
        except Exception as e:
            # 보간 실패시 가장 가까운 지점
            i = int(np.argmin(np.abs(fpr - fnr)))
            eer = float((fpr[i] + fnr[i]) / 2.0)
            tau = float(thr[i])
            print(f"⚠️ Interpolation failed, using closest point: tau={tau:.4f}, eer={eer:.4f}")
        
        # 🐋 범위 클리핑
        if self.clip_range is not None:
            tau = float(np.clip(tau, self.clip_range[0], self.clip_range[1]))
        
        # 🐋 통계 로깅
        stats = {
            'tau_raw': tau,
            'eer': eer,
            'genuine_mean': float(np.mean(genuine)),
            'genuine_std': float(np.std(genuine)),
            'impostor_mean': float(np.mean(impostor)),
            'impostor_std': float(np.std(impostor)),
            'separation': float(np.mean(genuine) - np.mean(impostor))
        }
        
        print(f"📊 EER Calculation:")
        print(f"   Genuine: {stats['genuine_mean']:.3f} ± {stats['genuine_std']:.3f}")
        print(f"   Impostor: {stats['impostor_mean']:.3f} ± {stats['impostor_std']:.3f}")
        print(f"   Separation: {stats['separation']:.3f}")
        print(f"   EER: {eer*100:.2f}%, Raw τ: {tau:.4f}")
        
        return tau, eer
    
    def smooth_tau(self, old_tau: Optional[float], new_tau: float) -> float:
        """
        🐋 EMA 스무딩 + 변화폭 제한
        
        Args:
            old_tau: 이전 임계치 (None이면 new_tau 그대로)
            new_tau: 새로 계산된 임계치
            
        Returns:
            스무딩된 임계치
        """
        if old_tau is None:
            # 첫 번째 계산
            tau = new_tau
        else:
            # EMA 적용
            tau = (1 - self.alpha) * old_tau + self.alpha * new_tau
            
            # 변화폭 제한
            delta = tau - old_tau
            if abs(delta) > self.max_delta:
                tau = old_tau + np.sign(delta) * self.max_delta
                print(f"⚠️ Delta clipped: {delta:.4f} → {np.sign(delta) * self.max_delta:.4f}")
        
        # 범위 클리핑
        if self.clip_range is not None:
            tau = float(np.clip(tau, self.clip_range[0], self.clip_range[1]))
        
        return float(tau)
    
    def auto_tune_margin(self, far_current: float) -> float:
        """
        🐋 (옵션) 마진 자동 조정
        
        Args:
            far_current: 현재 측정된 FAR
            
        Returns:
            조정된 마진값
        """
        if not self.use_auto_margin or self.far_target is None:
            return self.tau_m
        
        old_margin = self.tau_m
        
        if far_current > self.far_target:
            # FAR이 목표보다 높음 → 마진 증가 (더 엄격하게)
            self.tau_m = min(self.margin_bounds[1], self.tau_m + self.margin_step_up)
        elif far_current < self.far_target * 0.5:
            # FAR이 목표의 절반 이하 → 마진 감소 (덜 엄격하게)
            self.tau_m = max(self.margin_bounds[0], self.tau_m - self.margin_step_down)
        
        if self.tau_m != old_margin:
            print(f"🎯 Margin auto-tuned: {old_margin:.3f} → {self.tau_m:.3f} (FAR: {far_current:.3f})")
        
        return self.tau_m
    
    def calibrate(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                  old_tau: Optional[float] = None) -> Dict:
        """
        🐋 전체 캘리브레이션 프로세스
        
        Args:
            genuine_scores: genuine 점수들
            impostor_scores: impostor 점수들
            old_tau: 이전 임계치
            
        Returns:
            캘리브레이션 결과 딕셔너리
        """
        # 1. EER 임계치 계산
        tau_new, eer = self.compute_eer_threshold(genuine_scores, impostor_scores)
        
        # 2. 스무딩 적용
        tau_smoothed = self.smooth_tau(old_tau, tau_new)
        
        # 3. 현재값 업데이트
        self.tau_s_current = tau_smoothed
        
        # 4. 결과 기록
        result = {
            'tau_raw': tau_new,
            'tau_smoothed': tau_smoothed,
            'tau_old': old_tau if old_tau is not None else 0.7,
            'eer': eer,
            'tau_m': self.tau_m,
            'genuine_count': len(genuine_scores),
            'impostor_count': len(impostor_scores),
            'genuine_mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
            'impostor_mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0,
            'separation': float(np.mean(genuine_scores) - np.mean(impostor_scores)) if len(genuine_scores) > 0 and len(impostor_scores) > 0 else 0
        }
        
        # 5. 히스토리에 추가
        self.history.append(result)
        
        # 6. 결과 출력
        print(f"\n✅ Calibration Complete:")
        print(f"   τ_s: {old_tau:.4f} → {tau_new:.4f} → {tau_smoothed:.4f} (smoothed)")
        print(f"   EER: {eer*100:.2f}%")
        print(f"   Margin: τ_m = {self.tau_m:.3f}")
        
        return result
    
    def get_stats(self) -> Dict:
        """
        🐋 현재 상태 및 통계 반환
        """
        if not self.history:
            return {
                'tau_s': self.tau_s_current if self.tau_s_current else 0.7,
                'tau_m': self.tau_m,
                'calibrations': 0
            }
        
        latest = self.history[-1]
        return {
            'tau_s': latest['tau_smoothed'],
            'tau_m': self.tau_m,
            'eer': latest['eer'],
            'calibrations': len(self.history),
            'separation': latest['separation'],
            'genuine_mean': latest['genuine_mean'],
            'impostor_mean': latest['impostor_mean']
        }
    
    def reset(self):
        """🐋 히스토리 리셋"""
        self.history = []
        self.tau_s_current = None
        self.tau_m = self.margin_init if hasattr(self, 'margin_init') else 0.05


# 🐋 헬퍼 함수들
def compute_far_frr(tau: float, genuine_scores: np.ndarray, 
                    impostor_scores: np.ndarray) -> Tuple[float, float]:
    """
    🐋 주어진 임계치에서 FAR/FRR 계산
    
    Returns:
        (FAR, FRR)
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0, 0.0
    
    # FRR: genuine을 거부한 비율
    frr = np.mean(genuine_scores < tau)
    
    # FAR: impostor를 수락한 비율
    far = np.mean(impostor_scores >= tau)
    
    return float(far), float(frr)


def find_threshold_at_far(target_far: float, genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray) -> float:
    """
    🐋 목표 FAR에서의 임계치 찾기
    
    Args:
        target_far: 목표 FAR (예: 0.01 = 1%)
        
    Returns:
        해당 FAR을 달성하는 임계치
    """
    # impostor 점수를 정렬
    sorted_impostor = np.sort(impostor_scores)[::-1]  # 내림차순
    
    # target_far 위치의 점수
    idx = int(len(sorted_impostor) * target_far)
    idx = min(idx, len(sorted_impostor) - 1)
    
    tau = sorted_impostor[idx]
    
    return float(tau)


# 🐋 테스트 코드
if __name__ == "__main__":
    print("=== Threshold Calibrator Test ===\n")
    
    # 캘리브레이터 생성
    calibrator = ThresholdCalibrator(
        mode="cosine",
        alpha=0.2,
        max_delta=0.03,
        clip_range=(-1.0, 1.0),
        use_auto_margin=False,
        margin_init=0.05
    )
    
    # 가짜 점수 생성
    genuine = np.random.normal(0.8, 0.1, 100)  # 높은 점수
    impostor = np.random.normal(0.3, 0.15, 200)  # 낮은 점수
    
    # 첫 번째 캘리브레이션
    print("=== First Calibration ===")
    result1 = calibrator.calibrate(genuine, impostor, old_tau=0.7)
    
    # 두 번째 캘리브레이션 (스무딩 효과 확인)
    print("\n=== Second Calibration ===")
    genuine2 = np.random.normal(0.75, 0.12, 100)
    impostor2 = np.random.normal(0.35, 0.15, 200)
    result2 = calibrator.calibrate(genuine2, impostor2, old_tau=result1['tau_smoothed'])
    
    # FAR/FRR 테스트
    print("\n=== FAR/FRR Test ===")
    far, frr = compute_far_frr(result2['tau_smoothed'], genuine2, impostor2)
    print(f"At τ={result2['tau_smoothed']:.4f}: FAR={far*100:.2f}%, FRR={frr*100:.2f}%")
    
    # 목표 FAR 임계치
    tau_at_1pct = find_threshold_at_far(0.01, genuine2, impostor2)
    print(f"Threshold at FAR=1%: {tau_at_1pct:.4f}")