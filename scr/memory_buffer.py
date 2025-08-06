import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

class ReservoirSamplingBuffer:  # 🌕 Avalanche 기반
    """Buffer updated with reservoir sampling."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer_data = []  # (image, label) 튜플 리스트
        self.buffer_weights = []  # reservoir sampling weights
        
    def update(self, new_data: List[Tuple], new_labels: List[int]):
        """Update buffer with new data using reservoir sampling."""
        # 새 데이터에 random weights 할당
        new_weights = [random.random() for _ in new_data]
        
        # 기존 버퍼와 결합
        combined_data = self.buffer_data + list(zip(new_data, new_labels))
        combined_weights = self.buffer_weights + new_weights
        
        # weight로 정렬하여 상위 max_size개 선택
        sorted_indices = sorted(range(len(combined_weights)), 
                              key=lambda i: combined_weights[i], 
                              reverse=True)
        
        # 상위 max_size개만 유지
        sorted_indices = sorted_indices[:self.max_size]
        self.buffer_data = [combined_data[i] for i in sorted_indices]
        self.buffer_weights = [combined_weights[i] for i in sorted_indices]
    
    def sample(self, n: int) -> Tuple[List, List]:
        """Sample n items from buffer."""
        if len(self.buffer_data) == 0:
            return [], []
        
        n = min(n, len(self.buffer_data))
        indices = random.sample(range(len(self.buffer_data)), n)
        
        sampled_data = []
        sampled_labels = []
        for i in indices:
            data, label = self.buffer_data[i]
            sampled_data.append(data)
            sampled_labels.append(label)
            
        return sampled_data, sampled_labels
    
    def get_all(self) -> Tuple[List, List]:
        """Get all data from buffer."""
        if not self.buffer_data:
            return [], []
        
        all_data = []
        all_labels = []
        for data, label in self.buffer_data:
            all_data.append(data)
            all_labels.append(label)
            
        return all_data, all_labels


class ClassBalancedBuffer:  # 🌕 Avalanche 기반 + 🐣 최소 보장 추가
    """Stores samples for replay, equally divided over classes with minimum guarantee."""
    
    def __init__(self, max_size: int, min_samples_per_class: int = 10):
        self.max_size = max_size
        self.min_samples_per_class = min_samples_per_class
        self.buffer_groups: Dict[int, ReservoirSamplingBuffer] = {}
        self.seen_classes = set()
        
    def _get_class_sizes(self) -> Dict[int, int]:
        """Calculate size allocation for each class."""
        num_classes = len(self.seen_classes)
        if num_classes == 0:
            return {}
        
        # 최소 보장 후 남은 공간
        guaranteed_size = num_classes * self.min_samples_per_class
        remaining_size = self.max_size - guaranteed_size
        
        if remaining_size < 0:
            # 메모리가 부족한 경우 균등 분배
            equal_size = self.max_size // num_classes
            return {c: equal_size for c in self.seen_classes}
        
        # 최소 보장 + 남은 공간 균등 분배
        extra_per_class = remaining_size // num_classes
        remainder = remaining_size % num_classes
        
        sizes = {}
        for i, class_id in enumerate(sorted(self.seen_classes)):
            sizes[class_id] = self.min_samples_per_class + extra_per_class
            if i < remainder:
                sizes[class_id] += 1
                
        return sizes
    
    def update_from_dataset(self, data_list: List, label_list: List):
        """Update buffer with new data."""
        # 클래스별로 데이터 분리
        class_data: Dict[int, List] = defaultdict(list)
        for data, label in zip(data_list, label_list):
            class_data[label].append(data)
        
        # 새로운 클래스 추가
        self.seen_classes.update(class_data.keys())
        
        # 각 클래스별 크기 계산
        class_sizes = self._get_class_sizes()
        
        # 각 클래스별로 업데이트
        for class_id, data in class_data.items():
            if class_id not in self.buffer_groups:
                self.buffer_groups[class_id] = ReservoirSamplingBuffer(
                    class_sizes[class_id]
                )
            
            # Reservoir sampling으로 업데이트
            self.buffer_groups[class_id].update(
                data, [class_id] * len(data)
            )
            
            # 크기 조정
            self.buffer_groups[class_id].max_size = class_sizes[class_id]
        
        # 기존 클래스들의 크기도 조정
        for class_id in self.seen_classes:
            if class_id in self.buffer_groups:
                old_buffer = self.buffer_groups[class_id]
                old_buffer.max_size = class_sizes[class_id]
                
                # 크기가 줄어든 경우 데이터 잘라내기
                if len(old_buffer.buffer_data) > old_buffer.max_size:
                    old_buffer.buffer_data = old_buffer.buffer_data[:old_buffer.max_size]
                    old_buffer.buffer_weights = old_buffer.buffer_weights[:old_buffer.max_size]
    
    def sample(self, n: int) -> Tuple[List, List]:
        """Sample n items uniformly from all classes."""
        if not self.buffer_groups:
            return [], []
        
        # 각 클래스에서 균등하게 샘플링
        samples_per_class = n // len(self.buffer_groups)
        remainder = n % len(self.buffer_groups)
        
        all_data = []
        all_labels = []
        
        for i, (class_id, buffer) in enumerate(self.buffer_groups.items()):
            n_samples = samples_per_class
            if i < remainder:
                n_samples += 1
                
            data, labels = buffer.sample(n_samples)
            all_data.extend(data)
            all_labels.extend(labels)
        
        # 섞기
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        
        shuffled_data = [all_data[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        
        return shuffled_data, shuffled_labels
    
    def get_all_data(self) -> Tuple[List, List]:
        """Get all data from buffer."""
        all_data = []
        all_labels = []
        
        for buffer in self.buffer_groups.values():
            data, labels = buffer.get_all()
            all_data.extend(data)
            all_labels.extend(labels)
            
        return all_data, all_labels
    
    def __len__(self):
        return sum(len(buffer.buffer_data) for buffer in self.buffer_groups.values())


# 테스트용 코드 🐣
if __name__ == "__main__":
    # ClassBalancedBuffer 테스트
    buffer = ClassBalancedBuffer(max_size=100, min_samples_per_class=10)
    
    # 첫 번째 클래스 추가
    buffer.update_from_dataset(['img1', 'img2', 'img3'], [0, 0, 0])
    print(f"After class 0: {len(buffer)} samples")
    
    # 두 번째 클래스 추가
    buffer.update_from_dataset(['img4', 'img5'], [1, 1])
    print(f"After class 1: {len(buffer)} samples")
    
    # 샘플링 테스트
    sampled_data, sampled_labels = buffer.sample(20)
    print(f"Sampled {len(sampled_data)} items")
    print(f"Labels: {sampled_labels}")