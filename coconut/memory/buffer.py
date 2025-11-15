"""
Memory Buffer for COCONUT
Reservoir Sampling & Class-Balanced Buffers for Continual Learning
"""

from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import torch


class ReservoirSamplingBuffer:
    """
    Buffer updated with reservoir sampling.

    Reservoir Sampling은 전체 데이터 크기를 모르는 상황에서도
    모든 샘플이 동일한 확률로 선택되도록 보장하는 알고리즘입니다.
    """

    def __init__(self, max_size: int):
        """
        :param max_size: 버퍼에 저장할 수 있는 최대 샘플 수
        """
        super().__init__()
        self.max_size = max_size
        self._buffer_weights = torch.zeros(0)
        self.buffer = []  # (data, label) 튜플 리스트

    def update_from_dataset(self, new_data: List, new_labels: List):
        """
        새로운 데이터로 버퍼를 업데이트합니다.

        :param new_data: 새로운 데이터 리스트 (이미지 경로 또는 텐서)
        :param new_labels: 새로운 레이블 리스트
        """
        # 새 데이터에 random weight 할당 (0~1 사이의 값)
        new_weights = torch.rand(len(new_data))

        # 데이터와 레이블을 튜플로 묶어서 저장
        new_buffer = list(zip(new_data, new_labels))
        combined_buffer = self.buffer + new_buffer

        # 기존 weights와 새 weights 결합
        cat_weights = torch.cat([self._buffer_weights, new_weights])
        # weight 기준으로 내림차순 정렬
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        # 상위 max_size개만 선택
        buffer_idxs = sorted_idxs[: self.max_size]
        # 선택된 인덱스에 해당하는 데이터만 유지
        self.buffer = [combined_buffer[i] for i in buffer_idxs.tolist()]
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, new_size: int):
        """
        버퍼의 최대 크기를 변경합니다.
        크기가 줄어든 경우, weight가 높은 순서로 데이터를 유지합니다.

        :param new_size: 새로운 최대 크기
        """
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        # 이미 정렬되어 있으므로 앞에서부터 자르면 됨
        self.buffer = self.buffer[: self.max_size]
        self._buffer_weights = self._buffer_weights[: self.max_size]

    def sample(self, n: int) -> Tuple[List, List]:
        """
        버퍼에서 n개의 샘플을 무작위로 선택합니다.

        :param n: 선택할 샘플 수
        :return: (데이터 리스트, 레이블 리스트) 튜플
        """
        if not self.buffer:
            return [], []

        # 요청된 수와 버퍼 크기 중 작은 값 선택
        n = min(n, len(self.buffer))
        # 무작위 순열 생성 (중복 없이 샘플링)
        indices = torch.randperm(len(self.buffer))[:n].tolist()

        sampled_data = []
        sampled_labels = []
        for i in indices:
            data, label = self.buffer[i]
            sampled_data.append(data)
            sampled_labels.append(label)

        return sampled_data, sampled_labels


class ClassBalancedBuffer:
    """
    클래스별로 균등하게 샘플을 저장하는 버퍼입니다.

    각 클래스마다 독립적인 ReservoirSamplingBuffer를 유지하여,
    모든 클래스가 메모리에서 공평하게 표현되도록 합니다.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: Optional[int] = None,
        min_samples_per_class: int = 10
    ):
        """
        :param max_size: 전체 버퍼의 최대 용량
        :param adaptive_size: True면 새로운 클래스가 추가될 때마다
                             각 클래스의 할당량을 동적으로 조정
        :param total_num_classes: adaptive_size가 False일 때,
                                 미리 알고 있는 전체 클래스 수
        :param min_samples_per_class: 각 클래스별 최소 샘플 수 보장
        """
        self.max_size = max_size
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.min_samples_per_class = min_samples_per_class
        self.seen_classes: Set[int] = set()  # 지금까지 본 클래스들
        self.buffer_groups: Dict[int, ReservoirSamplingBuffer] = {}  # 클래스별 버퍼

    def get_group_lengths(self, num_groups):
        """
        각 클래스(그룹)에 할당될 버퍼 크기를 계산합니다.

        :param num_groups: 현재까지 본 클래스 수
        :return: 각 클래스별 할당 크기 리스트
        """
        if self.adaptive_size:
            # 최소 보장 계산
            guaranteed_size = num_groups * self.min_samples_per_class
            if guaranteed_size > self.max_size:
                # 메모리가 부족한 경우: 균등 분배 (최소 보장 포기)
                return [self.max_size // num_groups for _ in range(num_groups)]

            # 최소 보장 후 남은 공간을 균등 분배
            remaining = self.max_size - guaranteed_size
            base_size = remaining // num_groups
            extra = remaining % num_groups  # 나머지는 앞쪽 클래스에 1개씩 추가

            lengths = []
            for i in range(num_groups):
                size = self.min_samples_per_class + base_size
                if i < extra:
                    size += 1
                lengths.append(size)
            return lengths
        else:
            # Fixed size: 전체 클래스 수로 균등 분배
            lengths = [
                self.max_size // self.total_num_classes for _ in range(num_groups)
            ]
        return lengths

    def update_from_dataset(self, new_data: List, new_labels: List):
        """
        새로운 데이터로 버퍼를 업데이트합니다.

        :param new_data: 새로운 데이터 리스트
        :param new_labels: 새로운 레이블 리스트
        """
        if len(new_data) == 0:
            return

        # 클래스별로 데이터 인덱스 수집
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(new_labels):
            # tensor일 경우를 대비해 int로 변환
            label = int(label)
            cl_idxs[label].append(idx)

        # 클래스별로 데이터 분리
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            # 해당 클래스의 데이터와 레이블 추출
            cl_datasets[c] = ([new_data[i] for i in c_idxs],
                             [new_labels[i] for i in c_idxs])

        # 새로 본 클래스 추가
        self.seen_classes.update(cl_datasets.keys())

        # 각 클래스별 할당 크기 계산
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        # 클래스 ID 순서대로 크기 할당 (일관성을 위해 정렬)
        for class_id, ll in zip(sorted(self.seen_classes), lens):
            class_to_len[class_id] = ll

        # 각 클래스별로 버퍼 업데이트
        for class_id, (data_c, labels_c) in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                # 기존 클래스: 버퍼 업데이트
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(data_c, labels_c)
                old_buffer_c.resize(ll)
            else:
                # 새 클래스: 버퍼 생성
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(data_c, labels_c)
                self.buffer_groups[class_id] = new_buffer

        # 모든 버퍼의 크기 재조정 (새 클래스 추가로 인한 재분배)
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(class_to_len[class_id])

    def sample(self, n: int) -> Tuple[List, List]:
        """
        모든 클래스에서 균등하게 n개의 샘플을 선택합니다.

        :param n: 선택할 총 샘플 수
        :return: (데이터 리스트, 레이블 리스트) 튜플
        """
        if not self.buffer_groups:
            return [], []

        # 각 클래스에서 뽑을 샘플 수 계산
        samples_per_class = n // len(self.buffer_groups)
        remainder = n % len(self.buffer_groups)

        all_data = []
        all_labels = []

        for i, (class_id, buffer) in enumerate(self.buffer_groups.items()):
            n_samples = samples_per_class
            # 나머지를 앞쪽 클래스에 1개씩 추가
            if i < remainder:
                n_samples += 1

            data, labels = buffer.sample(n_samples)
            all_data.extend(data)
            all_labels.extend(labels)

        # 클래스 순서가 예측 가능하지 않도록 섞기
        if all_data:
            indices = torch.randperm(len(all_data)).tolist()
            shuffled_data = [all_data[i] for i in indices]
            shuffled_labels = [all_labels[i] for i in indices]
            return shuffled_data, shuffled_labels

        return all_data, all_labels

    def get_all_data(self) -> Tuple[List, List]:
        """버퍼의 모든 데이터를 반환"""
        all_data = []
        all_labels = []

        for buffer in self.buffer_groups.values():
            for data, label in buffer.buffer:
                all_data.append(data)
                all_labels.append(label)

        return all_data, all_labels

    def __len__(self):
        """버퍼에 저장된 총 샘플 수를 반환합니다."""
        return sum(len(buffer.buffer) for buffer in self.buffer_groups.values())
