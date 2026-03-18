"""
IDL + RTM Open-Set Loss for COCONUT

논문 아이디어 기반:
- IDL (Identification Loss): mated probe가 gallery에서 올바른 class를 top-r 안에 찾도록 soft rank 최소화
- Detection Loss: genuine score가 impostor threshold 후보보다 높도록 유지
- RTM (Relative Threshold Minimization): non-mated probe의 높은 impostor score를 직접 억제
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDL_RTMLoss(nn.Module):
    """
    Open-set training loss combining:
    - L_det: Detection loss (genuine similarity > threshold candidates)
    - L_id:  Identification loss (soft rank minimization via logsumexp)
    - L_rtm: Relative Threshold Minimization (suppress high impostor scores)

    배치 내부에서 post-hoc episode splitting으로 gallery/mated-probe/non-mated-probe 역할을 할당.
    """

    def __init__(self,
                 alpha_det: float = 1.0,
                 beta_id: float = 1.0,
                 gamma_rtm: float = 0.5,
                 n_quantiles: int = 5,
                 rank_target: float = 1.0,
                 rank_gamma: float = 1.0,
                 rank_temperature: float = 0.5,
                 rtm_temperature: float = 0.1,
                 det_steepness: float = 10.0,
                 min_gallery_classes: int = 2,
                 gallery_fraction: float = 0.7):
        super().__init__()
        self.alpha_det = alpha_det
        self.beta_id = beta_id
        self.gamma_rtm = gamma_rtm
        self.n_quantiles = n_quantiles
        self.rank_target = rank_target
        self.rank_gamma = rank_gamma
        self.rank_temperature = rank_temperature
        self.rtm_temperature = rtm_temperature
        self.det_steepness = det_steepness
        self.min_gallery_classes = min_gallery_classes
        self.gallery_fraction = gallery_fraction

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2-정규화된 임베딩
            labels: [B] 클래스 레이블

        Returns:
            loss: 스칼라 텐서
            info: dict (L_det, L_id, L_rtm, n_gallery, n_nonmated, skipped)
        """
        B, D = features.shape
        device = features.device

        # 1. 클래스별 인덱스 수집
        unique_classes = labels.unique()
        C = len(unique_classes)

        if C < self.min_gallery_classes:
            return torch.tensor(0.0, device=device, requires_grad=True), {'skipped': True}

        class_indices = {}
        for c in unique_classes:
            idx = (labels == c).nonzero(as_tuple=True)[0]
            class_indices[c.item()] = idx

        # 2. gallery 후보: ≥2 샘플인 클래스
        eligible = [c for c, idx in class_indices.items() if len(idx) >= 2]

        if len(eligible) < self.min_gallery_classes:
            return torch.tensor(0.0, device=device, requires_grad=True), {'skipped': True}

        # 3. gallery / non-mated 분할
        n_gallery = max(self.min_gallery_classes,
                        int(self.gallery_fraction * len(eligible)))
        # non-mated가 최소 1개 클래스 보장
        n_gallery = min(n_gallery, len(eligible) - 1)

        perm = torch.randperm(len(eligible))
        gallery_classes = [eligible[i] for i in perm[:n_gallery]]
        nonmated_classes = [eligible[i] for i in perm[n_gallery:]]

        # ≥2 샘플 미달 클래스도 non-mated에 추가
        ineligible = [c for c in class_indices if c not in eligible]
        nonmated_classes.extend(ineligible)

        # 4. gallery / mated probe 인덱스 할당
        gallery_idx_list = []
        mated_idx_list = []
        for c in gallery_classes:
            idx = class_indices[c]
            sel = idx[torch.randperm(len(idx), device=device)[:2]]
            gallery_idx_list.append(sel[0])
            mated_idx_list.append(sel[1])

        gallery_idx = torch.stack(gallery_idx_list)   # [G]
        mated_idx = torch.stack(mated_idx_list)       # [G]

        # 5. non-mated probe 인덱스
        nonmated_idx_list = []
        for c in nonmated_classes:
            nonmated_idx_list.append(class_indices[c])

        if nonmated_idx_list:
            nonmated_idx = torch.cat(nonmated_idx_list)
        else:
            nonmated_idx = torch.tensor([], dtype=torch.long, device=device)

        # 6. Similarity matrix
        S = features @ features.T  # [B, B]

        # 7. 세 손실 항 계산
        L_det = self._compute_det_loss(S, gallery_idx, mated_idx, nonmated_idx)
        L_id = self._compute_id_loss(S, gallery_idx, mated_idx)
        L_rtm = self._compute_rtm_loss(S, gallery_idx, nonmated_idx)

        # 8. 결합
        loss = self.alpha_det * L_det + self.beta_id * L_id + self.gamma_rtm * L_rtm

        info = {
            'L_det': L_det.item(),
            'L_id': L_id.item(),
            'L_rtm': L_rtm.item(),
            'n_gallery': len(gallery_classes),
            'n_nonmated': len(nonmated_idx),
            'skipped': False,
        }

        return loss, info

    def _compute_det_loss(self, S, gallery_idx, mated_idx, nonmated_idx):
        """Detection Loss: genuine score가 impostor threshold 후보보다 높도록"""
        device = S.device

        if len(nonmated_idx) == 0:
            return torch.tensor(0.0, device=device)

        # Genuine similarities
        s_genuine = S[mated_idx, gallery_idx]  # [G]

        # Non-mated probe → gallery similarities
        s_impostor = S[nonmated_idx][:, gallery_idx]  # [N, G]

        # Flatten impostor scores
        s_imp_flat = s_impostor.reshape(-1)  # [N*G]

        if len(s_imp_flat) < 2:
            return torch.tensor(0.0, device=device)

        # 미분 가능한 soft quantile: sort 후 quantile 위치 주변 window의 mean
        sorted_scores, _ = s_imp_flat.sort(descending=True)
        n_scores = len(sorted_scores)

        quantile_levels = torch.linspace(0.9, 0.99, self.n_quantiles, device=device)
        tau_candidates = []
        for q in quantile_levels:
            # 상위 (1-q) 비율 위치
            k = max(0, int((1.0 - q.item()) * n_scores))
            # 주변 window로 smooth quantile
            window_start = max(0, k - 2)
            window_end = min(n_scores, k + 3)
            tau_k = sorted_scores[window_start:window_end].mean()
            tau_candidates.append(tau_k)

        tau = torch.stack(tau_candidates)  # [K]

        # Detection loss: sigmoid(steepness * (τ_k - s_genuine))
        # s_genuine: [G], tau: [K] → broadcast: [G, K]
        margins = tau.unsqueeze(0) - s_genuine.unsqueeze(1)  # [G, K]
        L_det = torch.sigmoid(self.det_steepness * margins).mean()

        return L_det

    def _compute_id_loss(self, S, gallery_idx, mated_idx):
        """Identification Loss: 정답 gallery가 top-1에 오도록 soft rank 최소화 (logsumexp 안정화)"""
        device = S.device
        G = len(gallery_idx)

        if G < 2:
            return torch.tensor(0.0, device=device)

        # Mated probe → gallery similarity
        s_probe_gallery = S[mated_idx][:, gallery_idx]  # [G, G]
        s_genuine_diag = s_probe_gallery.diag()          # [G]

        # 대각선 제외 (자기 자신 제외)
        mask = ~torch.eye(G, dtype=torch.bool, device=device)
        s_others = s_probe_gallery[mask].view(G, G - 1)  # [G, G-1]

        # rank_temperature로 스케일링
        diff = (s_others - s_genuine_diag.unsqueeze(1)) / self.rank_temperature  # [G, G-1]

        # logsumexp로 안정적 계산: log(1 + Σ exp(diff))
        # = logsumexp([0, diff_1, diff_2, ...])
        zeros = torch.zeros(G, 1, device=device)
        log_soft_rank = torch.logsumexp(torch.cat([zeros, diff], dim=1), dim=1)  # [G]

        # Rank loss: softplus((log_soft_rank - log(r_target)) / γ)
        log_target = math.log(max(self.rank_target, 1e-6))
        L_id = F.softplus(
            (log_soft_rank - log_target) / self.rank_gamma
        ).mean()

        return L_id

    def _compute_rtm_loss(self, S, gallery_idx, nonmated_idx):
        """RTM: non-mated probe의 높은 impostor score를 softmax-weighted로 억제"""
        device = S.device

        if len(nonmated_idx) == 0:
            return torch.tensor(0.0, device=device)

        # Non-mated probe → gallery similarities
        s_nm = S[nonmated_idx][:, gallery_idx]  # [N, G]

        # Softmax attention weights (높은 score에 집중)
        w = F.softmax(s_nm / self.rtm_temperature, dim=1)  # [N, G]

        # Weighted similarity 합계의 평균
        L_rtm = (w * s_nm).sum(dim=1).mean()

        return L_rtm
