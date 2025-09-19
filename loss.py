"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 🪵 기존: 양성 0일 때 NaN 발생
        
        # 🌽 양성 0 앵커 안전 처리
        pos_per_row = mask.sum(1)  # 🌽 [B_total]
        valid = pos_per_row > 0    # 🌽 양성 있는 앵커만
        num = (mask * log_prob).sum(1)  # 🌽
        
        mean_log_prob_pos = torch.zeros_like(pos_per_row)  # 🌽
        mean_log_prob_pos[valid] = num[valid] / pos_per_row[valid].clamp_min(1)  # 🌽

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # 🪵
        # loss = loss.view(anchor_count, batch_size).mean()  # 🪵 reshape 제거 (valid 인덱싱 때문)
        
        # 🌽 valid 앵커만 손실 계산
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid]  # 🌽
        loss = loss.mean() if valid.any() else torch.zeros([], device=device)  # 🌽

        return loss

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