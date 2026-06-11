"""
Cohort recall-interval scheduler for COCONUT (MRS component ⓑ).

목적(role): 매 step 모든 옛 클래스를 균등 replay 하는 대신, 등록된 클래스를
phase 로 나눠 *간격(spacing)* 을 두고 recall 한다(spacing-effect 동기, plan G13/G15).
한 step 의 메모리 예산을 더 적은 클래스에 집중 → 클래스당 더 깊은 rehearsal,
그러나 각 클래스는 ``recall_interval`` step 마다 한 번은 반드시 recall.

At-risk 클래스(component ⓒ predictive override)는 이 phase 와 무관하게 항상 끼워
넣어야 하므로, union 은 trainer 가 수행한다(이 스케줄러는 phase-eligible 집합만 반환).

Default-OFF 규약:
    - ``recall_interval <= 1``  → eligible = None (= 모든 클래스, 현행 uniform replay)
    - 등록 클래스 수 < ``warmup_classes`` → eligible = None
  두 경우 모두 ``ClassBalancedBuffer.sample(n, eligible_class_ids=None)`` 의 현행
  경로를 타므로 byte-identical.
"""

from typing import List, Optional, Set


class CohortScheduler:
    """등록 순서를 phase 로 분할해 step 마다 recall 대상 클래스를 고른다.

    Parameters
    ----------
    recall_interval : int
        각 클래스가 recall 되는 주기 R. ``R <= 1`` 이면 off(전 클래스 매 step).
    warmup_classes : int
        등록 클래스가 이 수 미만이면 off(초기에는 전부 replay).
    """

    def __init__(self, recall_interval: int = 1, warmup_classes: int = 0):
        self.recall_interval = int(recall_interval)
        self.warmup_classes = int(warmup_classes)
        # 등록 순서(enrollment order)를 자체 유지 — trainer 의 registered_users 는
        # set(순서 없음)이므로 의존하지 않는다.
        self._order: List[int] = []
        self._seen: Set[int] = set()

    @property
    def enabled(self) -> bool:
        return self.recall_interval > 1

    def register(self, class_id: int) -> None:
        """클래스 등록 시 호출. 중복은 무시(등록 순서 보존)."""
        if class_id not in self._seen:
            self._seen.add(class_id)
            self._order.append(class_id)

    def reset(self) -> None:
        self._order.clear()
        self._seen.clear()

    def eligible(self, step: int) -> Optional[Set[int]]:
        """이 step 에서 recall 할 클래스 집합. off 면 None.

        phase = ``step % R`` 의 클래스(등록 index 기준 ``idx % R == phase``)만 반환.
        R step 을 돌면 모든 클래스가 정확히 한 번 recall 된다(union over phases = 전체).

        ⚠️ **계약(load-bearing): ``step`` 은 반드시 *experience index*
        (= ``trainer.experience_count``)여야 한다.** spacing 은 등록(experience)
        단위 간격이지 epoch/iteration 단위가 아니다. ``sample()`` 은 한 experience
        안에서 epoch×iteration 만큼 여러 번 호출되므로(trainer.py:611/614),
        ``iteration`` 을 넘기면 한 experience 안에서 phase 가 돌아 spacing 이
        무력화된다. experience 단위로 넘기면 eligible 집합이 그 experience 동안
        고정되어 (R-1)/R 의 클래스가 그 experience 에서 0회 rehearsal — 이 공백은
        의도된 것이며 component ⓒ(predictive override)가 floor 통과 전에 메운다.
        """
        if not self.enabled:
            return None
        if len(self._order) < self.warmup_classes:
            return None
        R = self.recall_interval
        phase = step % R
        return {cid for idx, cid in enumerate(self._order) if idx % R == phase}
