# coconut/training/evaluator.py
"""
Continual Learning 평가 메트릭 계산
- Average Accuracy
- Forgetting Measure
- Open-set metrics (TAR, TRR, FAR)
"""

import numpy as np
from collections import defaultdict
from typing import Dict


class ContinualLearningEvaluator:
    """
    Continual Learning 평가 메트릭 계산
    - Average Accuracy
    - Forgetting Measure
    - Open-set metrics (TAR, TRR, FAR)
    """
    def __init__(self, num_experiences: int):
        self.num_experiences = num_experiences
        self.accuracy_history = defaultdict(list)
        self.openset_history = []

    def update(self, experience_id: int, accuracy: float):
        """experience_id 학습 후 정확도 업데이트"""
        self.accuracy_history[experience_id].append(accuracy)

    def update_openset(self, metrics: dict):
        """오픈셋 메트릭 업데이트"""
        self.openset_history.append(metrics)

    def get_average_accuracy(self) -> float:
        """현재까지의 평균 정확도"""
        all_accs = []
        for accs in self.accuracy_history.values():
            if accs:
                all_accs.append(accs[-1])
        return np.mean(all_accs) if all_accs else 0.0

    def get_forgetting_measure(self) -> float:
        """Forgetting Measure 계산"""
        forgetting = []
        for exp_id, accs in self.accuracy_history.items():
            if len(accs) > 1:
                max_acc = max(accs[:-1])
                curr_acc = accs[-1]
                forgetting.append(max_acc - curr_acc)

        return np.mean(forgetting) if forgetting else 0.0

    def get_latest_openset_metrics(self) -> dict:
        """최신 오픈셋 메트릭 반환"""
        if self.openset_history:
            return self.openset_history[-1]
        return {}
