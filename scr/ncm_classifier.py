# 🌕 Avalanche의 ncm_classifier.py 기반

from typing import Dict
import torch
from torch import Tensor, nn


class NCMClassifier(nn.Module):  # 🌕 Avalanche 기반 (DynamicModule 의존성만 제거)
    """
    NCM (Nearest Class Mean) Classifier.
    
    각 클래스의 평균 feature vector (prototype)를 저장하고,
    새로운 입력이 들어오면 가장 가까운 클래스로 분류합니다.
    
    🌕 Avalanche의 NCMClassifier를 최대한 유지
    """

    def __init__(self, normalize: bool = True):
        """
        :param normalize: 입력을 L2 정규화할지 여부.
                         True면 cosine similarity 기반 분류
                         False면 Euclidean distance 기반 분류
        """
        super().__init__()
        # 🌕 Avalanche와 동일한 구조
        self.register_buffer("class_means", None)  # [num_classes, feature_size]
        self.class_means_dict = {}  # {class_id: mean_vector}
        
        self.normalize = normalize
        self.max_class = -1

    def load_state_dict(self, state_dict, strict: bool = True):  # 🌕 Avalanche 그대로
        """
        체크포인트에서 상태를 로드합니다.
        클래스 평균을 복원하는 데 필수적입니다.
        """
        self.class_means = state_dict["class_means"]
        super().load_state_dict(state_dict, strict)
        # 텐서에서 딕셔너리 재구성
        if self.class_means is not None:
            for i in range(self.class_means.shape[0]):
                if (self.class_means[i] != 0).any():
                    self.class_means_dict[i] = self.class_means[i].clone()
        self.max_class = max(self.class_means_dict.keys()) if self.class_means_dict else -1

    def _vectorize_means_dict(self):  # 🌕 Avalanche 그대로
        """
        딕셔너리 형태의 class means를 텐서로 변환합니다.
        빠른 거리 계산을 위해 필요합니다.
        """
        if self.class_means_dict == {}:
            return

        max_class = max(self.class_means_dict.keys())
        self.max_class = max(max_class, self.max_class)
        
        # 첫 번째 mean vector로 feature 차원 확인
        first_mean = list(self.class_means_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device
        
        # 모든 클래스를 담을 수 있는 텐서 생성
        self.class_means = torch.zeros(self.max_class + 1, feature_size).to(device)

        # 딕셔너리에서 텐서로 복사
        for k, v in self.class_means_dict.items():
            self.class_means[k] = self.class_means_dict[k].clone()

    @torch.no_grad()
    def forward(self, x):  # 🌕 Avalanche 그대로
        """
        입력 x에 대해 각 클래스까지의 거리를 계산합니다.
        
        :param x: (batch_size, feature_size)
        :return: (batch_size, num_classes) - 각 클래스까지의 negative distance
        """
        if self.class_means_dict == {}:
            # 초기화되지 않은 경우 처리
            self.init_missing_classes(range(self.max_class + 1), x.shape[1], x.device)

        assert self.class_means_dict != {}, "no class means available."
        
        if self.normalize:
            # L2 정규화 (cosine similarity를 위해)
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        # 모든 클래스 평균과의 거리 계산
        # (num_classes, batch_size)
        sqd = torch.cdist(self.class_means.to(x.device), x)
        
        # negative distance 반환 (값이 클수록 가까움)
        # (batch_size, num_classes)
        return (-sqd).T

    def update_class_means_dict(
        self, class_means_dict: Dict[int, Tensor], momentum: float = 0.5  # 🌕 Avalanche 기본값
    ):
        """
        클래스 평균을 업데이트합니다.
        
        🌕 Avalanche의 기본값 0.5 유지
        - momentum = 0.5: 이전 지식과 새 지식을 동등하게 가중
        - momentum = 1.0: 완전 교체 (이전 지식 무시)
        - momentum = 0.0: 업데이트 안함 (새 지식 무시)
        
        Continual learning에서는 0.5가 catastrophic forgetting을
        완화하는 데 효과적입니다.
        
        :param class_means_dict: {클래스 ID: 평균 벡터} 딕셔너리
        :param momentum: 새로운 평균의 가중치 (0.0 ~ 1.0)
        """
        assert momentum <= 1 and momentum >= 0
        assert isinstance(class_means_dict, dict), (
            "class_means_dict must be a dictionary mapping class_id " "to mean vector"
        )
        
        for k, v in class_means_dict.items():
            if k not in self.class_means_dict or (self.class_means_dict[k] == 0).all():
                # 새로운 클래스
                self.class_means_dict[k] = class_means_dict[k].clone()
            else:
                # 기존 클래스 업데이트 (momentum 적용)
                device = self.class_means_dict[k].device
                self.class_means_dict[k] = (
                    momentum * class_means_dict[k].to(device)
                    + (1 - momentum) * self.class_means_dict[k]
                )

        self._vectorize_means_dict()

    def replace_class_means_dict(self, class_means_dict: Dict[int, Tensor]):  # 🌕 Avalanche 그대로
        """
        기존 평균을 완전히 교체합니다.
        momentum = 1.0과 동일한 효과입니다.
        """
        assert isinstance(class_means_dict, dict), (
            "class_means_dict must be a dictionary mapping class_id " "to mean vector"
        )
        self.class_means_dict = {k: v.clone() for k, v in class_means_dict.items()}
        self._vectorize_means_dict()

    def init_missing_classes(self, classes, class_size, device):  # 🌕 Avalanche 그대로
        """
        아직 평균이 없는 클래스를 0 벡터로 초기화합니다.
        """
        for k in classes:
            if k not in self.class_means_dict:
                self.class_means_dict[k] = torch.zeros(class_size).to(device)
        self._vectorize_means_dict()

    def adaptation(self, experience):  # 🌕 Avalanche의 adaptation (단순화)
        """
        새로운 experience에 맞춰 모델을 적응시킵니다.
        
        새로운 클래스가 나타나면 자동으로 처리합니다.
        
        :param experience: 현재 experience (classes_in_this_experience 필요)
        """
        # 🔄 DynamicModule의 super().adaptation() 제거
        
        if hasattr(experience, 'classes_in_this_experience'):
            classes = experience.classes_in_this_experience
            for k in classes:
                self.max_class = max(k, self.max_class)
            
            if self.class_means is not None:
                self.init_missing_classes(
                    classes, self.class_means.shape[1], self.class_means.device
                )
    
    # 🐣 추가 편의 메서드들
    def predict(self, x):
        """
        실제 클래스 예측을 반환합니다.
        
        :param x: (batch_size, feature_size)
        :return: (batch_size,) - 예측된 클래스 ID
        """
        scores = self.forward(x)  # (batch_size, num_classes)
        return scores.argmax(dim=1)
    
    def get_num_classes(self):
        """현재 저장된 클래스 수를 반환합니다."""
        return len(self.class_means_dict)
    
    def get_class_means(self):
        """현재 저장된 클래스 평균들을 반환합니다."""
        return self.class_means_dict.copy()


# 🐣 테스트 코드
if __name__ == "__main__":
    # NCM Classifier 테스트
    ncm = NCMClassifier(normalize=True)
    
    # 가상의 클래스 평균 설정
    class_means = {
        0: torch.randn(128).float(),  # 클래스 0의 평균
        1: torch.randn(128).float(),  # 클래스 1의 평균
    }
    
    # momentum 테스트
    print("=== Momentum 테스트 ===")
    ncm.update_class_means_dict(class_means, momentum=0.5)  # 기본값
    
    # 같은 클래스에 새로운 평균 업데이트
    new_means = {
        0: torch.randn(128).float(),  # 클래스 0의 새 평균
    }
    ncm.update_class_means_dict(new_means, momentum=0.3)  # 30%만 반영
    
    # 테스트 입력
    test_input = torch.randn(5, 128)  # 5개 샘플, 128차원
    
    # 예측
    predictions = ncm.predict(test_input)
    print(f"Predictions: {predictions}")
    print(f"Number of classes: {ncm.get_num_classes()}")