"""
Herding Buffer for COCONUT
iCaRL-inspired representative sample selection for improved continual learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from PIL import Image

from .buffer import ClassBalancedBuffer, ReservoirSamplingBuffer


class HerdingBuffer(ClassBalancedBuffer):
    """
    Herding 전략을 사용하는 메모리 버퍼.
    각 클래스의 평균 특징에 가장 가까운 샘플들을 선택하여 저장합니다.

    iCaRL (Rebuffi et al., CVPR 2017) 논문의 herding 전략 기반
    """

    def __init__(
        self,
        max_size: int,
        model: nn.Module,
        device: torch.device,
        transform=None,
        adaptive_size: bool = True,
        min_samples_per_class: int = 5,
        max_samples_per_class: int = 20,
        use_projection: bool = False
    ):
        """
        Args:
            max_size: 전체 버퍼의 최대 용량
            model: 특징 추출용 모델
            device: 연산 디바이스
            transform: 이미지 전처리 함수
            adaptive_size: 동적 크기 조정 여부
            min_samples_per_class: 클래스당 최소 샘플 수
            max_samples_per_class: 클래스당 최대 샘플 수 (herding용)
            use_projection: projection head 사용 여부
        """
        super().__init__(
            max_size=max_size,
            adaptive_size=adaptive_size,
            min_samples_per_class=min_samples_per_class
        )

        self.model = model
        self.device = device
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.use_projection = use_projection

        # Drift 관련 파라미터 저장
        self.drift_threshold = 0.5  # 기본값, config에서 override 가능

        # 클래스별 특징 평균 저장
        self.class_means: Dict[int, torch.Tensor] = {}

        # 클래스별 대표 샘플과 특징 저장
        self.herding_samples: Dict[int, List[Tuple[str, torch.Tensor]]] = {}

        # Feature drift 모니터링용 통계
        self.class_stats: Dict[int, Dict] = {}

    @torch.no_grad()
    def extract_features(self, paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        이미지 경로 리스트에서 특징 벡터를 추출합니다.

        Args:
            paths: 이미지 경로 리스트
            batch_size: 배치 크기

        Returns:
            특징 벡터 텐서 (N, D)
        """
        self.model.eval()
        features_list = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_images = []

            # 이미지 로드 및 전처리
            for path in batch_paths:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    batch_images.append(img)

            # 배치 텐서 생성
            batch_tensor = torch.stack(batch_images).to(self.device)

            # 특징 추출
            features = self.model(batch_tensor)

            # Projection head 사용 여부
            if self.use_projection and hasattr(self.model, 'projection_head'):
                features = self.model.projection_head(features)
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            features_list.append(features.cpu())

        return torch.cat(features_list, dim=0) if features_list else torch.empty(0)

    def herding_select(self, paths: List[str], labels: List[int], n_per_class: int) -> Dict[int, List[str]]:
        """
        Herding 알고리즘으로 각 클래스별 대표 샘플을 선택합니다.

        Args:
            paths: 전체 이미지 경로 리스트
            labels: 레이블 리스트
            n_per_class: 클래스당 선택할 샘플 수

        Returns:
            클래스별 선택된 경로 딕셔너리
        """
        # 클래스별로 데이터 분리
        class_data = defaultdict(list)
        for path, label in zip(paths, labels):
            class_data[int(label)].append(path)

        selected_samples = {}

        for class_id, class_paths in class_data.items():
            if len(class_paths) == 0:
                continue

            # 현재 클래스의 모든 샘플에 대한 특징 추출
            features = self.extract_features(class_paths)

            # 클래스 평균 계산
            class_mean = features.mean(dim=0, keepdim=True)
            self.class_means[class_id] = class_mean.squeeze()

            # Herding: 평균에 가장 가까운 샘플부터 선택
            selected_indices = []
            selected_features = []

            current_mean = torch.zeros_like(class_mean)

            for _ in range(min(n_per_class, len(class_paths))):
                # 현재까지 선택된 샘플들의 평균
                if selected_features:
                    current_mean = torch.stack(selected_features).mean(dim=0, keepdim=True)

                # 다음 샘플 선택: (현재평균 + 새샘플) / n이 클래스평균에 가장 가까운 것
                remaining_indices = [i for i in range(len(class_paths)) if i not in selected_indices]

                if not remaining_indices:
                    break

                # 각 후보 샘플을 추가했을 때의 새 평균 계산
                min_distance = float('inf')
                best_idx = -1

                for idx in remaining_indices:
                    # 새 평균 계산
                    new_features = selected_features + [features[idx]]
                    new_mean = torch.stack(new_features).mean(dim=0)

                    # 클래스 평균과의 거리
                    distance = torch.norm(new_mean - class_mean.squeeze())

                    if distance < min_distance:
                        min_distance = distance
                        best_idx = idx

                selected_indices.append(best_idx)
                selected_features.append(features[best_idx])

            # 선택된 샘플 경로 저장
            selected_samples[class_id] = [class_paths[i] for i in selected_indices]

            # 선택된 샘플과 특징 저장
            self.herding_samples[class_id] = [
                (class_paths[i], features[i]) for i in selected_indices
            ]

            # 통계 업데이트 (drift 모니터링용)
            self.class_stats[class_id] = {
                'mean': class_mean.squeeze(),
                'std': features.std(dim=0),
                'num_samples': len(selected_indices)
            }

        return selected_samples

    def update_from_dataset(self, new_data: List, new_labels: List, adaptive: bool = True):
        """
        Herding 전략을 사용하여 버퍼를 업데이트합니다.
        Adaptive 모드에서는 drift가 감지된 클래스만 재선택합니다.

        Args:
            new_data: 새로운 데이터 경로 리스트
            new_labels: 새로운 레이블 리스트
            adaptive: True면 drift 감지 시에만 재선택 (효율적)
        """
        if len(new_data) == 0:
            return

        # 클래스별로 데이터 분리
        class_data = defaultdict(list)
        for path, label in zip(new_data, new_labels):
            class_data[int(label)].append(path)

        # Adaptive update: drift 감지 후 선택적 재선택
        if adaptive and len(self.herding_samples) > 0:
            classes_to_update = set()
            drift_count = 0

            # 새 데이터가 있는 클래스의 drift 검사
            for class_id, paths in class_data.items():
                if len(paths) == 0:
                    continue

                # 새 데이터의 특징 추출
                new_features = self.extract_features(paths[:min(10, len(paths))])  # 효율성을 위해 일부만 검사

                # Drift 감지
                if class_id not in self.class_stats:
                    # 새 클래스는 무조건 추가
                    classes_to_update.add(class_id)
                elif self.detect_drift(class_id, new_features, threshold=getattr(self, 'drift_threshold', 0.5)):
                    classes_to_update.add(class_id)
                    drift_count += 1
                    print(f"[Adaptive] Drift detected for class {class_id}")

            if drift_count > 0:
                print(f"[Adaptive] {drift_count} classes need update due to drift")

            # Drift 감지된 클래스와 새 클래스만 재선택
            if classes_to_update:
                # 기존 샘플과 새 샘플 합치기
                combined_data = defaultdict(list)

                # 기존 샘플 유지 (업데이트 불필요한 클래스)
                final_paths = []
                final_labels = []

                for class_id, samples in self.herding_samples.items():
                    if class_id not in classes_to_update:
                        # Drift 없음 - 기존 샘플 그대로 유지
                        for path, _ in samples:
                            final_paths.append(path)
                            final_labels.append(class_id)
                    else:
                        # Drift 있음 - 재선택 대상에 추가
                        for path, _ in samples:
                            combined_data[class_id].append(path)

                # 새 샘플 추가 (재선택 대상 클래스)
                for class_id in classes_to_update:
                    if class_id in class_data:
                        combined_data[class_id].extend(class_data[class_id])

                # 재선택 대상 클래스만 herding 수행
                if combined_data:
                    all_paths = []
                    all_labels = []

                    for class_id, paths in combined_data.items():
                        all_paths.extend(paths)
                        all_labels.extend([class_id] * len(paths))

                    num_classes = len(combined_data)
                    samples_per_class = min(
                        self.max_samples_per_class,
                        max(
                            self.min_samples_per_class,
                            self.max_size // (len(self.herding_samples) + len(classes_to_update))
                            if len(self.herding_samples) > 0 else self.max_samples_per_class
                        )
                    )

                    # Herding 선택
                    selected_samples = self.herding_select(all_paths, all_labels, samples_per_class)

                    # 재선택된 샘플 추가
                    for class_id, paths in selected_samples.items():
                        final_paths.extend(paths)
                        final_labels.extend([class_id] * len(paths))

                    print(f"[Adaptive] Re-selected {len(final_paths) - sum(1 for c in self.herding_samples.keys() if c not in classes_to_update)} samples from {num_classes} classes")

            else:
                # Drift 없음 - 기존 샘플 유지, 새 샘플만 추가
                print(f"[Adaptive] No drift detected, keeping existing samples")
                final_paths = []
                final_labels = []

                for class_id, samples in self.herding_samples.items():
                    for path, _ in samples:
                        final_paths.append(path)
                        final_labels.append(class_id)

                # 새 클래스만 추가 (기존 클래스에는 drift 없음)
                for class_id, paths in class_data.items():
                    if class_id not in self.herding_samples:
                        combined_paths = paths
                        n_samples = min(self.max_samples_per_class, len(combined_paths))
                        selected = self.herding_select(combined_paths, [class_id] * len(combined_paths), n_samples)

                        if class_id in selected:
                            final_paths.extend(selected[class_id])
                            final_labels.extend([class_id] * len(selected[class_id]))

        else:
            # Non-adaptive: 전체 재선택 (첫 업데이트 또는 adaptive=False)
            combined_data = defaultdict(list)
            combined_labels = defaultdict(list)

            # 기존 herding 샘플 추가
            for class_id, samples in self.herding_samples.items():
                for path, _ in samples:
                    combined_data[class_id].append(path)
                    combined_labels[class_id].append(class_id)

            # 새 샘플 추가
            for class_id, paths in class_data.items():
                combined_data[class_id].extend(paths)
                combined_labels[class_id].extend([class_id] * len(paths))

            # 전체 데이터에서 herding으로 선택
            all_paths = []
            all_labels = []

            for class_id in combined_data.keys():
                all_paths.extend(combined_data[class_id])
                all_labels.extend(combined_labels[class_id])

            # 클래스당 샘플 수 계산
            num_classes = len(combined_data)
            samples_per_class = min(
                self.max_samples_per_class,
                max(
                    self.min_samples_per_class,
                    self.max_size // num_classes if num_classes > 0 else self.max_samples_per_class
                )
            )

            # Herding 선택 수행
            selected_samples = self.herding_select(all_paths, all_labels, samples_per_class)

            # 부모 클래스의 버퍼 업데이트
            final_paths = []
            final_labels = []

            for class_id, paths in selected_samples.items():
                final_paths.extend(paths)
                final_labels.extend([class_id] * len(paths))

            print(f"[Herding] Full update: {len(final_paths)} samples from {num_classes} classes")

        # ClassBalancedBuffer 업데이트
        super().update_from_dataset(final_paths, final_labels)

        print(f"[Herding] Buffer updated: {len(final_paths)} total samples")

    def detect_drift(self, class_id: int, new_features: torch.Tensor, threshold: float = 0.5) -> bool:
        """
        특정 클래스의 feature drift를 감지합니다.

        Args:
            class_id: 클래스 ID
            new_features: 새로운 특징 벡터들
            threshold: drift 판단 임계값

        Returns:
            drift 발생 여부
        """
        if class_id not in self.class_stats:
            return False

        old_mean = self.class_stats[class_id]['mean']
        new_mean = new_features.mean(dim=0)

        # 평균 간 거리 계산 (L2 norm)
        drift_score = torch.norm(new_mean - old_mean).item()

        # 표준편차 고려한 정규화
        old_std = self.class_stats[class_id]['std']
        normalized_drift = drift_score / (old_std.mean().item() + 1e-6)

        return normalized_drift > threshold

    def get_drift_statistics(self) -> Dict[int, float]:
        """
        모든 클래스의 drift 통계를 반환합니다.

        Returns:
            클래스별 drift score 딕셔너리
        """
        drift_scores = {}

        for class_id, samples in self.herding_samples.items():
            if class_id not in self.class_stats:
                continue

            # 현재 저장된 샘플들의 특징
            current_features = torch.stack([feat for _, feat in samples])
            current_mean = current_features.mean(dim=0)

            # 원래 평균과의 차이
            original_mean = self.class_stats[class_id]['mean']
            drift_score = torch.norm(current_mean - original_mean).item()

            drift_scores[class_id] = drift_score

        return drift_scores

    def adaptive_update(self, class_id: int, new_paths: List[str], new_labels: List[int]):
        """
        Drift가 감지된 클래스만 선택적으로 업데이트합니다.

        Args:
            class_id: 업데이트할 클래스 ID
            new_paths: 새 샘플 경로들
            new_labels: 새 샘플 레이블들
        """
        # 새 특징 추출
        new_features = self.extract_features(new_paths)

        # Drift 감지
        if self.detect_drift(class_id, new_features):
            print(f"[Herding] Drift detected for class {class_id}, updating samples...")

            # 기존 샘플과 합치기
            if class_id in self.herding_samples:
                old_paths = [path for path, _ in self.herding_samples[class_id]]
                combined_paths = old_paths + new_paths
                combined_labels = [class_id] * len(combined_paths)
            else:
                combined_paths = new_paths
                combined_labels = new_labels

            # Re-select using herding
            n_samples = min(self.max_samples_per_class, len(combined_paths))
            selected = self.herding_select(combined_paths, combined_labels, n_samples)

            # 해당 클래스만 업데이트
            if class_id in selected:
                super().update_from_dataset(selected[class_id], [class_id] * len(selected[class_id]))