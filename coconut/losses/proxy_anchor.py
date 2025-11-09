"""
ProxyAnchorLoss for COCONUT
분리된 독립 모듈로 구성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)

    프록시를 앵커로 사용하여 data-to-data relations를 활용하는 손실 함수
    Continual Learning을 위한 동적 프록시 추가 지원
    """
    def __init__(self, embedding_size=128, margin=0.1, alpha=32):
        super().__init__()
        self.embedding_size = embedding_size
        self.margin = margin
        self.alpha = alpha

        # 동적 프록시 관리
        self.proxies = None
        self.class_to_idx = {}
        self.num_classes = 0

    @torch.no_grad()
    def add_classes(self, class_ids):
        """새 클래스 프록시 추가 (gradient 계산 없음)"""
        new_ids = [c for c in class_ids if c not in self.class_to_idx]
        if not new_ids:
            return

        n_new = len(new_ids)
        device = (self.proxies.device if self.proxies is not None
                  else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 새 프록시 초기화 (kaiming_normal)
        new_p = torch.randn(n_new, self.embedding_size, device=device)
        nn.init.kaiming_normal_(new_p, mode='fan_out')

        if self.proxies is None:
            self.proxies = nn.Parameter(new_p)
        else:
            combined = torch.cat([self.proxies.data, new_p], dim=0)
            self.proxies = nn.Parameter(combined)

        # 매핑 업데이트
        for i, cid in enumerate(new_ids):
            self.class_to_idx[cid] = self.num_classes + i
        self.num_classes += n_new

        print(f"ProxyAnchor: Added {n_new} proxies, Total: {self.num_classes}")

    def forward(self, X, T):
        """
        Args:
            X: [B, D] 임베딩 벡터
            T: [B] 레이블
        Returns:
            스칼라 손실값
        """
        # 프록시 없으면 0 반환 (디바이스 안전)
        if self.proxies is None:
            return torch.zeros([], device=X.device, dtype=X.dtype)

        # L2 정규화
        X = F.normalize(X, p=2, dim=1)
        P = F.normalize(self.proxies, p=2, dim=1)

        # 코사인 유사도 (명확한 @ 연산)
        sim = X @ P.T  # [B, C]

        # 벡터화된 one-hot 생성
        labels = T.to(dtype=torch.long, device=sim.device)
        known_mask = torch.tensor(
            [int(y.item()) in self.class_to_idx for y in labels],
            device=sim.device, dtype=torch.bool
        )

        # 알려진 클래스가 없으면 0 반환
        if not known_mask.any():
            return torch.zeros([], device=X.device, dtype=X.dtype)

        # 알려진 클래스만 처리
        valid_labels = labels[known_mask]
        valid_sim = sim[known_mask]  # [B', C]

        # 프록시 인덱스 매핑
        proxy_indices = torch.tensor(
            [self.class_to_idx[int(y.item())] for y in valid_labels],
            device=sim.device, dtype=torch.long
        )

        # One-hot 인코딩
        P_one_hot = F.one_hot(proxy_indices, num_classes=self.num_classes).to(valid_sim.dtype)
        N_one_hot = 1 - P_one_hot

        # Exponentials (수치 안정성)
        pos_exp = torch.exp(-self.alpha * (valid_sim - self.margin))
        neg_exp = torch.exp(self.alpha * (valid_sim + self.margin))

        # dim=0 집계 - 프록시가 앵커!
        P_sim_sum = (pos_exp * P_one_hot).sum(dim=0)  # [C]
        N_sim_sum = (neg_exp * N_one_hot).sum(dim=0)  # [C]

        # Positive 항: 양성 샘플이 실제로 존재하는 프록시만
        with_pos = (P_one_hot.sum(dim=0) > 0)
        num_valid = with_pos.sum().clamp_min(1)

        # 수치 안정성: log1p + clamp_min
        pos_term = torch.log1p(P_sim_sum[with_pos].clamp_min(1e-12)).sum() / num_valid
        neg_term = torch.log1p(N_sim_sum.clamp_min(1e-12)).sum() / self.num_classes

        return pos_term + neg_term

    def get_proxies(self):
        """정규화된 프록시 반환"""
        if self.proxies is None:
            return None
        return F.normalize(self.proxies, p=2, dim=1)