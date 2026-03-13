# coconut/classifiers/threshold.py
"""
FAR 타겟 기반 임계치 계산 및 스무딩
- τ = quantile(unknown_dev_scores, 1 - target_FPIR)
- 코사인 유사도 기준
- EMA 스무딩 + 변화폭 제한
- 선택적 마진 자동 조정
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)  # interp1d 경고 무시


class ThresholdCalibrator:
    """
    전역 임계치(τ_s) 캘리브레이터
    - τ = quantile(unknown_dev_scores, 1 - target_FPIR)
    - 코사인 유사도 기반(기본)
    - EMA 스무딩 + 변화폭 제한
    - (옵션) 마진 자동 조정
    """

    def __init__(
        self,
        mode: str = "cosine",                    # "cosine" 또는 "euclidean"
        threshold_mode: str = "far",
        target_far: float = 0.01,                # FAR 타겟 (1%)
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
        verbose: bool = True,                    # ️ 상세 출력
    ):
        self.mode = mode
        self.threshold_mode = threshold_mode     # ️ EER or FAR
        self.target_far = target_far             # ️ FAR 타겟값
        self.alpha = alpha
        self.max_delta = max_delta
        self.clip_range = clip_range
        self.verbose = verbose                   #         
        #  마진 관련
        self.use_auto_margin = use_auto_margin
        self.tau_m = margin_init
        self.margin_bounds = margin_bounds
        self.margin_step_up = margin_step_up
        self.margin_step_down = margin_step_down
        #  self.far_target = far_target  # 중복 제거
        self.far_target_margin = far_target      # ️ 마진용 FAR 타겟 (이름 변경)
        
        #  최소 샘플 요구사항
        self.min_samples = min_samples
        
        #  히스토리 추적
        self.history: List[Dict] = []
        self.tau_s_current: Optional[float] = None
        
    def compute_far_threshold(self, impostor: np.ndarray) -> Tuple[float, float]:
        """
        ️ FAR 타겟 임계치 계산
        
        Args:
            impostor: impostor 페어의 유사도 점수
            
        Returns:
            (tau_new, achieved_far): 새로운 임계치와 실제 달성 FAR
        """
        if len(impostor) < self.min_samples:
            if self.verbose:
                print(f"WARNING: Not enough impostor samples: {len(impostor)}")
            return self.tau_s_current if self.tau_s_current else 0.7, 0.0
        
        # ️ FAR = P(impostor >= tau) = target_far
        # 따라서 tau = quantile(impostor, 1 - target_far)
        tau = np.quantile(impostor, 1 - self.target_far)
        
        # ️ 실제 달성된 FAR 계산
        achieved_far = np.mean(impostor >= tau)
        
        # ️ 범위 클리핑
        if self.clip_range is not None:
            tau = float(np.clip(tau, self.clip_range[0], self.clip_range[1]))
        
        if self.verbose:
            print(f"️ FAR Target Calculation:")
            print(f"   Target FAR: {self.target_far*100:.2f}%")
            print(f"   Achieved FAR: {achieved_far*100:.2f}%")
            print(f"   Threshold τ: {tau:.4f}")
            print(f"   Impostor stats: {np.mean(impostor):.3f} ± {np.std(impostor):.3f}")
        
        return float(tau), float(achieved_far)
    
    def smooth_tau(self, old_tau: Optional[float], new_tau: float) -> float:
        """
         EMA 스무딩 + 변화폭 제한 (기존 유지)
        
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
                if self.verbose:
                    print(f"WARNING: Delta clipped: {delta:.4f} → {np.sign(delta) * self.max_delta:.4f}")
        
        # 범위 클리핑
        if self.clip_range is not None:
            tau = float(np.clip(tau, self.clip_range[0], self.clip_range[1]))
        
        return float(tau)
    
    def auto_tune_margin(self, far_current: float) -> float:
        """
         (옵션) 마진 자동 조정 (기존 유지)
        
        Args:
            far_current: 현재 측정된 FAR
            
        Returns:
            조정된 마진값
        """
        if not self.use_auto_margin or self.far_target_margin is None:  # ️ 변수명 변경
            return self.tau_m
        
        old_margin = self.tau_m
        
        if far_current > self.far_target_margin:
            # FAR이 목표보다 높음 → 마진 증가 (더 엄격하게)
            self.tau_m = min(self.margin_bounds[1], self.tau_m + self.margin_step_up)
        elif far_current < self.far_target_margin * 0.5:
            # FAR이 목표의 절반 이하 → 마진 감소 (덜 엄격하게)
            self.tau_m = max(self.margin_bounds[0], self.tau_m - self.margin_step_down)
        
        if self.tau_m != old_margin and self.verbose:
            print(f"[TARGET] Margin auto-tuned: {old_margin:.3f} → {self.tau_m:.3f} (FAR: {far_current:.3f})")
        
        return self.tau_m
    
    def calibrate(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                  old_tau: Optional[float] = None) -> Dict:
        """
         전체 캘리브레이션 프로세스
        ️ FAR 타겟 모드 추가
        
        Args:
            genuine_scores: genuine 점수들
            impostor_scores: impostor 점수들
            old_tau: 이전 임계치
            
        Returns:
            캘리브레이션 결과 딕셔너리
        """
        tau_new, achieved_metric = self.compute_far_threshold(impostor_scores)
        metric_name = "far"
        
        # 2. 스무딩 적용
        tau_smoothed = self.smooth_tau(old_tau, tau_new)
        
        # 3. 현재값 업데이트
        self.tau_s_current = tau_smoothed
        
        # ️ 4. 현재 FAR/FRR 계산
        current_far = np.mean(impostor_scores >= tau_smoothed) if len(impostor_scores) > 0 else 0
        current_frr = np.mean(genuine_scores < tau_smoothed) if len(genuine_scores) > 0 else 0
        
        # 5. 결과 기록
        result = {
            'tau_raw': tau_new,
            'tau_smoothed': tau_smoothed,
            'tau_old': old_tau if old_tau is not None else 0.7,
            metric_name: achieved_metric,  # eer 또는 far
            'current_far': current_far,
            'current_frr': current_frr,
            'tau_m': self.tau_m,
            'genuine_count': len(genuine_scores),
            'impostor_count': len(impostor_scores),
            'genuine_mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
            'impostor_mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0,
            'separation': float(np.mean(genuine_scores) - np.mean(impostor_scores)) if len(genuine_scores) > 0 and len(impostor_scores) > 0 else 0,
            'mode': self.threshold_mode
        }
        
        # 6. 히스토리에 추가
        self.history.append(result)
        
        # 7. 결과 출력
        if self.verbose:
            print(f"\n[OK] Calibration Complete ({self.threshold_mode.upper()} mode):")
            
            # old_tau가 None일 수 있으므로 처리
            if old_tau is not None:
                print(f"   τ_s: {old_tau:.4f} → {tau_new:.4f} → {tau_smoothed:.4f} (smoothed)")
            else:
                print(f"   τ_s: initial → {tau_new:.4f} → {tau_smoothed:.4f} (smoothed)")
            
            print(f"   Target FPIR: {self.target_far*100:.2f}%")
            print(f"   Achieved FPIR: {achieved_metric*100:.2f}%")
            
            print(f"   Current FAR: {current_far*100:.2f}%, FRR: {current_frr*100:.2f}%")
            if self.use_auto_margin:
                print(f"   Margin: τ_m = {self.tau_m:.3f}")
        return result
    
    def get_stats(self) -> Dict:
        """
         현재 상태 및 통계 반환
        ️ FAR 모드 정보 추가
        """
        if not self.history:
            return {
                'tau_s': self.tau_s_current if self.tau_s_current else 0.7,
                'tau_m': self.tau_m,
                'calibrations': 0,
                'mode': self.threshold_mode
            }
        
        latest = self.history[-1]
        return {
            'tau_s': latest['tau_smoothed'],
            'tau_m': self.tau_m,
            'eer': latest.get('eer', None),  # ️ EER 모드일 때만
            'far': latest.get('far', None),  # ️ FAR 모드일 때만
            'current_far': latest.get('current_far', None),
            'current_frr': latest.get('current_frr', None),
            'calibrations': len(self.history),
            'separation': latest['separation'],
            'genuine_mean': latest['genuine_mean'],
            'impostor_mean': latest['impostor_mean'],
            'mode': self.threshold_mode
        }
    
    def reset(self):
        """ 히스토리 리셋"""
        self.history = []
        self.tau_s_current = None
        self.tau_m = self.margin_init if hasattr(self, 'margin_init') else 0.05


# ️ 테스트 코드 수정
if __name__ == "__main__":
    print("=== Threshold Calibrator Test ===\n")
    
    # ️ FAR 타겟 캘리브레이터 생성
    print("️ Testing FAR Target Mode")
    print("-" * 50)
    calibrator_far = ThresholdCalibrator(
        mode="cosine",
        threshold_mode="far",  # ️ FAR 타겟
        target_far=0.01,  # ️ 1% FAR
        alpha=0.2,
        max_delta=0.03,
        clip_range=(-1.0, 1.0),
        use_auto_margin=False,
        margin_init=0.05,
        verbose=True
    )
    
    # 가짜 점수 생성
    genuine = np.random.normal(0.8, 0.1, 100)  # 높은 점수
    impostor = np.random.normal(0.3, 0.15, 200)  # 낮은 점수
    
    # FAR 모드 테스트
    print("\n=== FAR Mode Calibration ===")
    result_far = calibrator_far.calibrate(genuine, impostor, old_tau=0.7)
    
    print("\n" + "="*50)
    print("FAR Mode Results")
    print("-" * 50)
    print(f"FAR Mode: τ={result_far['tau_smoothed']:.4f}, FPIR={result_far['current_far']*100:.2f}%, FNIR(FRR)={result_far['current_frr']*100:.2f}%")