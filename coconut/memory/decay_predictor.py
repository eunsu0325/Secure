"""
Predictive decay override for COCONUT (MRS component ⓒ).

목적(role): ``ForgettingTracker`` 의 per-user 성능 궤적을 외삽해, *다음* step 에
floor 아래로 떨어질 것으로 예측되는 클래스를 미리 recall 집합에 끼워 넣는다
(cohort phase 와 무관하게). Pavlik/ACT-R 동기이되, **하드코딩된 망각 상수 금지** —
오직 관측 궤적의 slope/EMA 만 사용한다(plan §H 가드레일).

핵심 함정(가드레일):
    - **floor = raw cosine 임계값** (same-domain impostor p99 등). NCM 의
      ``tau_cos`` 는 per-class z-score(S-norm) 이므로 floor 로 쓰면 안 된다.
      floor 값은 *호출자가* 주입한다(이 모듈은 메트릭 종류를 가정하지 않음).
    - **new-user 필터**: 평가 이력이 ``min_history`` 미만이면 slope 미정의 → override
      안 함(신규 등록자를 즉시 at-risk 로 오탐하지 않게).
    - **low-baseline 필터**: 한 번도 ``min_baseline`` 이상 못 간 클래스는 '망각'이
      아니라 애초에 약한 것 → 예산 낭비 방지 위해 제외.
    - **cap**: 한 step override 최대 수 제한(예산 폭주 방지). trainer 가 QAR 과
      union 한 뒤 다시 cap 을 적용한다(이중 예산 금지).

Default-OFF 규약: ``enabled=False`` → ``at_risk(...)`` 가 항상 빈 set 반환 →
trainer 의 eligible/override 가 변하지 않아 byte-identical.
"""

from typing import Iterable, List, Optional, Set


class DecayPredictor:
    """성능 궤적 외삽으로 at-risk 클래스를 예측한다.

    Parameters
    ----------
    enabled : bool
        False(기본)면 항상 빈 set 반환(off).
    metric : str
        ForgettingTracker 궤적에서 읽을 메트릭 키 (예: '1-eer', 'tar_001', rank-1).
        floor 와 단위가 일치해야 한다.
    higher_is_better : bool
        메트릭 방향. True(기본)=값이 클수록 좋음(TPIR/genuine/1-eer; floor 아래로
        떨어지면 at-risk). False=값이 작을수록 좋음('eer'; floor 위로 *올라가면*
        at-risk). ⚠️ tracker 에는 'eer'(lower-better)도 저장되므로 이 플래그를
        잘못 두면 로직이 *조용히* 반전된다 — 반드시 메트릭과 맞출 것.
    floor : float
        worsening 임계값. **raw cosine 기반 임계값을 주입**(NCM tau_cos 금지).
    horizon : int
        몇 step 앞을 외삽할지(slope * horizon).
    min_history : int
        slope 추정에 필요한 최소 평가 횟수. 미만이면 override 안 함(new-user 보호).
    min_baseline : Optional[float]
        low-baseline 필터. None(기본)=끔. 값이 주어지면, 궤적이 한 번도 'good'
        수준(higher_is_better=True: max>=baseline / False: min<=baseline)에
        도달 못 한 클래스를 제외(애초에 약한 것 ≠ 망각).
    slope_mode : str
        'linear'(최근 window 선형회귀 기울기) 또는 'ema'(연속 delta 의 EMA).
    window : int
        slope_mode='linear' 에서 사용할 최근 포인트 수(>=2).
    ema_alpha : float
        slope_mode='ema' 의 평활 계수(0<alpha<=1).
    cap : Optional[int]
        한 번에 반환할 최대 at-risk 클래스 수(None=무제한). 위험도(예측 마진)가
        낮은(=더 위험한) 순으로 cap.
    """

    def __init__(
        self,
        enabled: bool = False,
        metric: str = "1-eer",
        higher_is_better: bool = True,
        floor: float = 0.0,
        horizon: int = 1,
        min_history: int = 2,
        min_baseline: Optional[float] = None,
        slope_mode: str = "linear",
        window: int = 3,
        ema_alpha: float = 0.5,
        cap: Optional[int] = None,
    ):
        if slope_mode not in ("linear", "ema"):
            raise ValueError(f"unknown slope_mode: {slope_mode!r}")
        self.enabled = bool(enabled)
        self.metric = metric
        self.higher_is_better = bool(higher_is_better)
        self.floor = float(floor)
        self.horizon = int(horizon)
        self.min_history = max(2, int(min_history))
        self.min_baseline = None if min_baseline is None else float(min_baseline)
        self.slope_mode = slope_mode
        self.window = max(2, int(window))
        self.ema_alpha = float(ema_alpha)
        self.cap = cap

    # ------------------------------------------------------------------ #
    def at_risk(self, tracker, candidate_ids: Optional[Iterable[int]] = None) -> Set[int]:
        """floor 아래로 떨어질 것으로 예측되는 클래스 set 을 반환한다.

        Parameters
        ----------
        tracker : ForgettingTracker
            ``get_performance_trajectory(user_id, metric) -> List[float]`` 제공.
        candidate_ids : iterable, optional
            검사 대상 클래스. None 이면 tracker 에 기록된 모든 user.
        """
        if not self.enabled:
            return set()

        if candidate_ids is None:
            candidate_ids = list(tracker.performance_matrix.keys())

        scored = []  # (severity, class_id) — severity 작을수록 위험(sort asc)
        for cid in candidate_ids:
            traj = tracker.get_performance_trajectory(cid, self.metric)
            if traj is None or len(traj) < self.min_history:
                continue  # new-user 보호
            if self._below_baseline(traj):
                continue  # low-baseline 필터(애초에 약한 것)
            slope = self._slope(traj)
            predicted = traj[-1] + slope * self.horizon
            if self.higher_is_better:
                if slope >= 0:
                    continue           # 하락 중이 아니면 override 불필요
                if predicted < self.floor:
                    scored.append((predicted, cid))      # 낮을수록 위험
            else:
                if slope <= 0:
                    continue           # 악화(상승) 중이 아니면 불필요
                if predicted > self.floor:
                    scored.append((-predicted, cid))     # 높을수록 위험 → 부호반전

        scored.sort(key=lambda t: t[0])  # 가장 위험한 것 먼저
        if self.cap is not None:
            scored = scored[: self.cap]
        return {cid for _, cid in scored}

    def _below_baseline(self, traj: List[float]) -> bool:
        """한 번도 'good' 수준에 도달 못 했는가(direction-aware)."""
        if self.min_baseline is None:
            return False
        if self.higher_is_better:
            return max(traj) < self.min_baseline
        return min(traj) > self.min_baseline

    # ------------------------------------------------------------------ #
    def _slope(self, traj: List[float]) -> float:
        """관측 궤적의 최근 기울기(step 당 변화량). 하드코딩 상수 없음."""
        if self.slope_mode == "ema":
            deltas = [traj[i] - traj[i - 1] for i in range(1, len(traj))]
            if not deltas:
                return 0.0
            ema = deltas[0]
            for d in deltas[1:]:
                ema = self.ema_alpha * d + (1.0 - self.ema_alpha) * ema
            return ema
        # linear: 최근 window 포인트의 최소제곱 기울기
        w = traj[-self.window:]
        m = len(w)
        if m < 2:
            return 0.0
        xs = list(range(m))
        mean_x = sum(xs) / m
        mean_y = sum(w) / m
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, w))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den
