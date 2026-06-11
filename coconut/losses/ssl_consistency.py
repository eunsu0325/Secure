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


# plan §H 에서 부르던 이름과의 호환 별칭
VBMConsistencyLoss = SSLConsistencyLoss
