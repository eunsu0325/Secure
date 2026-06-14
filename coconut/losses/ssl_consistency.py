"""
SSL consistency loss for COCONUT (MRS component ⓐ).

목적(role): replay/recall 시점에 *retention(보존)* 을 강화하는 self-supervised
consistency 항. cross-domain genuine 붕괴를 *치료*하는 도구가 아니다(그건 augmentation
영역; plan G11/G13 참고). 두 augmented view 의 표현이 서로 일치하도록 정렬해
sequential 재학습 동안 표현이 덜 흔들리게 한다.

Repo-verified 전제 (coconut/training/trainer.py:670-704):
    view1 = data[0]; view2 = data[1]
    x = torch.cat([view1, view2], dim=0)            # [2B, ...]
    features_all = self.model(x)                    # [2B, D]  (projection space)
손실은 ``features_all[:B]`` (view1) 과 ``features_all[B:]`` (view2) 를 받아 계산한다.
=> 추가 forward 0회 (dual-view forward 를 그대로 재사용).

Default-OFF 규약: trainer 는 ``w_ssl == 0`` 일 때 이 모듈을 *호출하지 않는다* →
학습 경로가 현행과 byte-identical. (0 을 곱해도 forward 값은 같지만 grad graph 에
노드가 추가되므로, byte-identity 는 trainer 의 호출 가드로 보장한다.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLConsistencyLoss(nn.Module):
    """View-consistency regularizer for replayed/current samples.

    Modes
    -----
    ``"cosine"`` (default):
        ``1 - mean cos(view1, view2)``. 대칭(symmetric) — grad 가 두 view 모두에
        흐른다. prototype 불필요 → sampler/NCM 와 완전 독립.

    ``"proto_kl"``:
        각 view 의 (prototype 에 대한) soft-assignment 분포가 일치하도록 강제.
        ``logits_v = (view_v @ prototypes.T) / temperature`` 의 softmax 분포 간
        대칭 KL (= Jeffreys divergence). prototypes 가 주어지지 않으면 자동으로
        cosine 모드로 폴백한다(학습 초기 NCM class_means 미구축 상황 보호).

    Notes
    -----
    - 입력은 이미 projection space 의 [B, D] 텐서라고 가정한다. cosine 모드는
      내부에서 L2 정규화하므로 입력 정규화 여부와 무관하게 안정적이다.
    - 이 모듈은 *가중치를 곱하지 않는다*. trainer 가 ``w_ssl * ssl_loss(...)`` 로
      스케일한다(off 가드와 책임 분리).
    """

    def __init__(self, mode: str = "cosine", temperature: float = 0.1):
        super().__init__()
        if mode not in ("cosine", "proto_kl"):
            raise ValueError(f"unknown ssl mode: {mode!r}")
        self.mode = mode
        self.temperature = float(temperature)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor,
                prototypes: torch.Tensor = None) -> torch.Tensor:
        if view1.shape != view2.shape:
            raise ValueError(
                f"view shape mismatch: {tuple(view1.shape)} vs {tuple(view2.shape)}"
            )

        if self.mode == "proto_kl" and prototypes is not None and prototypes.numel() > 0:
            return self._proto_kl(view1, view2, prototypes)
        # cosine (default, and proto_kl 폴백)
        return self._cosine(view1, view2)

    def _cosine(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(view1, view2, dim=1)
        return (1.0 - cos).mean()

    def _proto_kl(self, view1: torch.Tensor, view2: torch.Tensor,
                  prototypes: torch.Tensor) -> torch.Tensor:
        protos = F.normalize(prototypes, dim=1)
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        logits1 = (z1 @ protos.t()) / self.temperature
        logits2 = (z2 @ protos.t()) / self.temperature
        logp1 = F.log_softmax(logits1, dim=1)
        logp2 = F.log_softmax(logits2, dim=1)
        p1 = logp1.exp()
        p2 = logp2.exp()
        # 대칭 KL (Jeffreys): KL(p1||p2) + KL(p2||p1)
        kl_12 = F.kl_div(logp2, p1, reduction="batchmean")
        kl_21 = F.kl_div(logp1, p2, reduction="batchmean")
        return 0.5 * (kl_12 + kl_21)

    # ------------------------------------------------------------------ #
    # AAVB C2 — VBM Eq.3 one-to-many divergence (plan §L C2, D6/D7)
    # ------------------------------------------------------------------ #
    def one_to_many(self, weak: torch.Tensor, strong_views, prototypes: torch.Tensor = None) -> torch.Tensor:
        """VBM Eq.3: ``L_ssl = 1/(V-1) Σ_j D_KL(p_weak.detach() ‖ p_strong_j)``.

        weak view(=anchor)의 soft-assignment 분포로 strong view들을 *끌어당긴다*
        (방향성). weak는 **detach** — anchor를 고정해 strong만 정렬(D6; detach 누락 시
        anchor도 strong 쪽으로 끌려가 메커니즘이 깨짐).

        Parameters
        ----------
        weak : Tensor [B, D]        — 약증강(anchor) view feature.
        strong_views : list[Tensor] — 강증강 view feature들 (V-1개). 비면 0 반환.
        prototypes : Tensor [C, D]  — 클래스 프로토타입(proxy). None/빈 텐서면
            cosine 폴백(strong을 weak.detach()로 끌어당김; D7).

        F.kl_div(input=log q, target=p) = KL(p‖q) 규약을 이용:
            KL(p_weak ‖ p_strong) = F.kl_div(log_p_strong, p_weak.detach()).
        """
        if strong_views is None or len(strong_views) == 0:
            return weak.new_zeros(())

        use_proto = (self.mode == "proto_kl"
                     and prototypes is not None and prototypes.numel() > 0)

        if use_proto:
            protos = F.normalize(prototypes, dim=1)
            logp_weak = F.log_softmax(
                (F.normalize(weak, dim=1) @ protos.t()) / self.temperature, dim=1)
            p_weak = logp_weak.exp().detach()      # anchor 고정 (D6)
            total = weak.new_zeros(())
            for s in strong_views:
                logp_s = F.log_softmax(
                    (F.normalize(s, dim=1) @ protos.t()) / self.temperature, dim=1)
                total = total + F.kl_div(logp_s, p_weak, reduction="batchmean")
            return total / len(strong_views)

        # cosine 폴백 (방향성: strong → weak.detach()). prototypes 없을 때(초기 등).
        w = F.normalize(weak, dim=1).detach()
        total = weak.new_zeros(())
        for s in strong_views:
            total = total + (1.0 - F.cosine_similarity(F.normalize(s, dim=1), w, dim=1)).mean()
        return total / len(strong_views)


# plan §H 에서 부르던 이름과의 호환 별칭
VBMConsistencyLoss = SSLConsistencyLoss
