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


# 💎 ProxyContrastLoss 추가 - 프로토타입 기반 대조 학습
class ProxyContrastLoss(nn.Module):
    """
    💎 Prototype-based Contrastive Loss for Continual Learning
    
    프로토타입(클래스 대표점)과의 대조 학습으로 catastrophic forgetting 방지
    - 고정된 앵커 역할로 드리프트 감소
    - Top-K 선택으로 계산 효율성 확보
    - 정답 클래스 강제 포함으로 안정성 보장
    
    Args:
        temperature: 소프트맥스 온도 (높을수록 부드러운 분포)
        lambda_proxy: 손실 가중치
        topk: 비교할 최대 클래스 수
        full_until: 이 수 이하면 모든 클래스 사용
    """
    def __init__(self, temperature=0.15, lambda_proxy=0.3, topk=30, full_until=100):
        super().__init__()
        self.temperature = temperature  # 💎 SupCon보다 높은 온도 (0.15 vs 0.07)
        self.lambda_proxy = lambda_proxy
        self.topk = topk
        self.full_until = full_until
    
    def forward(self, z, y, proto_cache_P, proto_cache_ids):
        """
        💎 Forward pass
        
        Args:
            z: [B, D] 배치 임베딩 (L2 정규화됨)
            y: [B] 라벨
            proto_cache_P: [C, D] 프로토타입 텐서 (L2 정규화됨)
            proto_cache_ids: 정렬된 클래스 ID 리스트
            
        Returns:
            스칼라 손실값
        """
        # 💎 프로토타입이 없으면 0 반환 (초기 상태)
        if proto_cache_P is None or len(proto_cache_ids) == 0:
            return torch.tensor(0.0, device=z.device)
        
        B = z.size(0)  # 배치 크기
        C = proto_cache_P.size(0)  # 클래스 수
        
        # 💎 클래스 ID → 인덱스 매핑
        id2idx = {cid: i for i, cid in enumerate(proto_cache_ids)}
        
        # 💎 코사인 유사도 계산 (이미 정규화된 벡터들)
        # AMP 대응
        if z.dtype == torch.float16:
            sim = (z.float() @ proto_cache_P.float().t()) / self.temperature
        else:
            sim = (z @ proto_cache_P.t()) / self.temperature  # [B, C]
        
        # 💎 Top-K 선택 또는 전체 사용
        if C <= self.full_until or self.topk >= C:
            # 클래스가 적으면 전체 사용
            sel_idx = torch.arange(C, device=z.device).unsqueeze(0).expand(B, -1)  # [B, C]
            sel_sim = sim
        else:
            # 💎 Top-K 클래스만 선택 (계산 효율성)
            sel_sim, sel_idx = sim.topk(self.topk, dim=1)  # [B, K]
            
            # 💎 정답 클래스 강제 포함 (중요!)
            for i in range(B):
                true_label = y[i].item()
                if true_label in id2idx:
                    true_idx = id2idx[true_label]
                    
                    # 💎 텐서 in 연산 버그 방지 - .any() 사용
                    exists = (sel_idx[i] == true_idx).any()
                    if not exists:
                        # Top-K에 없으면 마지막 자리를 정답으로 교체
                        sel_sim[i, -1] = sim[i, true_idx]
                        sel_idx[i, -1] = true_idx
        
        # 💎 Cross Entropy 손실 계산
        losses = []
        for i in range(B):
            true_label = y[i].item()
            
            # 프로토타입이 없는 새 클래스는 스킵
            if true_label not in id2idx:
                continue
            
            true_idx = id2idx[true_label]
            
            # 💎 선택된 클래스 중 정답 위치 찾기
            pos_mask = (sel_idx[i] == true_idx)
            if not pos_mask.any():
                continue  # 정답이 없으면 스킵 (방어 코드)
            
            pos_in_topk = pos_mask.nonzero(as_tuple=True)[0][0]
            
            # 💎 Softmax cross entropy
            log_probs = sel_sim[i].log_softmax(dim=0)  # [K or C]
            losses.append(-log_probs[pos_in_topk])
        
        # 💎 평균 손실 반환
        if losses:
            return self.lambda_proxy * torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=z.device)
    
    def update_lambda(self, new_lambda):
        """💎 Lambda 값 동적 업데이트 (스케줄링용)"""
        self.lambda_proxy = new_lambda