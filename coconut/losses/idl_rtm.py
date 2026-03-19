"""
IDL + RTM Open-Set Loss for COCONUT

"Open-Set Biometrics: Beyond Good Closed-Set Models" (Su et al., MSU)

논문 수식 충실 구현:
- Eq. 7-8: Detection Score S^det — non-mated score를 threshold로 사용
- Eq. 9-10: Identification Score S^id — sigmoid 기반 soft rank
- Eq. 11: L^IDL = -(1/|P'_K|) Σ S^det_i · S^id_i (곱 결합)
- Eq. 12: L^RTM = softmax-weighted max impostor score 억제
- Eq. 13: L = L^IDL + λ · L^RTM
"""

import torch
import torch.nn as nn


class IDL_RTMLoss(nn.Module):
    """
    Open-set training loss (Su et al.):
    - L^IDL: Joint detection-identification loss (Eq. 11)
    - L^RTM: Relative Threshold Minimization (Eq. 12)

    배치 내부에서 post-hoc episode splitting으로
    gallery / mated-probe / non-mated-probe 역할을 할당.
    """

    def __init__(self,
                 alpha: float = 6.0,          # Detection sigmoid steepness (Eq. 7)
                 beta: float = 0.2,           # Identification sigmoid steepness (Eq. 9)
                 gamma: float = 6.0,          # Rank sigmoid steepness (Eq. 10)
                 rtm_lambda: float = 4.0,     # λ: RTM weight (Eq. 13)
                 min_gallery_classes: int = 2,
                 gallery_fraction: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rtm_lambda = rtm_lambda
        self.min_gallery_classes = min_gallery_classes
        self.gallery_fraction = gallery_fraction

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2-정규화된 임베딩
            labels: [B] 클래스 레이블

        Returns:
            loss: 스칼라 텐서
            info: dict with L_IDL, L_RTM, S_det_mean, S_id_mean, etc.
        """
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
        n_gallery = min(n_gallery, len(eligible) - 1)

        perm = torch.randperm(len(eligible))
        gallery_classes = [eligible[i] for i in perm[:n_gallery]]
        nonmated_classes = [eligible[i] for i in perm[n_gallery:]]

        # ≥2 샘플 미달 클래스도 non-mated에 추가
        ineligible = [c for c in class_indices if c not in eligible]
        nonmated_classes.extend(ineligible)

        # 4. gallery / mated probe 인덱스 할당 (클래스당 2샘플 → 1 gallery, 1 mated probe)
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

        # 6. Similarity matrix (cosine sim, L2-norm 전제)
        S = features @ features.T  # [B, B]

        G = len(gallery_idx)

        # ===== L^IDL (Eq. 11): -(1/|P'_K|) Σ S^det_i · S^id_i =====
        S_det = self._compute_S_det(S, gallery_idx, mated_idx, nonmated_idx)  # [G]
        S_id = self._compute_S_id(S, gallery_idx, mated_idx)                   # [G]
        L_IDL = -(S_det * S_id).mean()

        # ===== L^RTM (Eq. 12) =====
        L_RTM = self._compute_rtm_loss(S, gallery_idx, nonmated_idx)

        # ===== Total (Eq. 13): L = L^IDL + λ · L^RTM =====
        loss = L_IDL + self.rtm_lambda * L_RTM

        info = {
            'L_IDL': L_IDL.item(),
            'L_RTM': L_RTM.item(),
            'S_det_mean': S_det.mean().item(),
            'S_id_mean': S_id.mean().item(),
            'n_gallery': G,
            'n_nonmated': len(nonmated_idx),
            'skipped': False,
        }

        return loss, info

    def _compute_S_det(self, S, gallery_idx, mated_idx, nonmated_idx):
        """
        Detection Score (Eq. 7-8):
        T = {s(n_j, g_k) : ∀j ∈ non-mated, ∀k ∈ gallery}
        S^det_i = (1/|T|) Σ_{t∈T} σ_α(s(p_i, g_i) - t)

        genuine score가 모든 impostor threshold보다 높으면 S^det → 1
        """
        device = S.device
        G = len(gallery_idx)

        if len(nonmated_idx) == 0:
            return torch.ones(G, device=device)

        # Genuine similarities: s(p_i, g_i)
        s_genuine = S[mated_idx, gallery_idx]  # [G]

        # Threshold set T: s(n_j, g_k) for all j, k
        s_impostor = S[nonmated_idx][:, gallery_idx]  # [N, G]
        T = s_impostor.reshape(-1)  # [N*G]

        # S^det_i = mean σ_α(s(p_i,g_i) - t) over all t
        # [G, 1] - [1, |T|] → [G, |T|]
        margins = s_genuine.unsqueeze(1) - T.unsqueeze(0)
        S_det = torch.sigmoid(self.alpha * margins).mean(dim=1)  # [G]

        return S_det

    def _compute_S_id(self, S, gallery_idx, mated_idx):
        """
        Identification Score (Eq. 9-10):
        softrank(p_i) = Σ_{j≠i} σ_γ(s(p_i, g_j) - s(p_i, g_i))   (Eq. 10)
        S^id_i = σ_β(1 - softrank(p_i) / (K-1))                    (Eq. 9)

        정답 gallery가 top-1이면 softrank→0, S^id→σ_β(1)≈1
        """
        device = S.device
        G = len(gallery_idx)

        if G < 2:
            return torch.ones(G, device=device)

        # Mated probe → all gallery similarities: s(p_i, g_j)
        s_probe_gallery = S[mated_idx][:, gallery_idx]  # [G, G]
        s_genuine_diag = s_probe_gallery.diag()          # [G] = s(p_i, g_i)

        # 대각선(자기 gallery) 제외
        mask = ~torch.eye(G, dtype=torch.bool, device=device)
        s_others = s_probe_gallery[mask].view(G, G - 1)  # [G, G-1] = s(p_i, g_j) j≠i

        # softrank(p_i) = Σ_{j≠i} σ_γ(s(p_i,g_j) - s(p_i,g_i))
        diff = s_others - s_genuine_diag.unsqueeze(1)  # [G, G-1]
        softrank = torch.sigmoid(self.gamma * diff).sum(dim=1)  # [G]

        # S^id_i = σ_β(1 - softrank/(K-1))
        K_minus_1 = max(G - 1, 1)
        S_id = torch.sigmoid(self.beta * (1.0 - softrank / K_minus_1))  # [G]

        return S_id

    def _compute_rtm_loss(self, S, gallery_idx, nonmated_idx):
        """
        RTM Loss (Eq. 12):
        L^RTM = (1/|N_K|) Σ_j w_j · max_i s(n_j, g_i)
        w_j = exp(max_i s(n_j, g_i)) / Σ_l exp(max_i s(n_l, g_i))

        높은 impostor score를 가진 non-mated probe에 집중하여 억제
        """
        device = S.device

        if len(nonmated_idx) == 0:
            return torch.tensor(0.0, device=device)

        # Non-mated probe → gallery similarities
        s_nm = S[nonmated_idx][:, gallery_idx]  # [N, G]

        # max_i s(n_j, g_i) per non-mated probe
        max_scores, _ = s_nm.max(dim=1)  # [N]

        # Softmax weights (논문: temperature 없음)
        w = torch.softmax(max_scores, dim=0)  # [N]

        # L^RTM = (1/|N_K|) Σ_j w_j · max_scores_j
        N_K = len(nonmated_idx)
        L_RTM = (w * max_scores).sum() / N_K

        return L_RTM
