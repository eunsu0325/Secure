"""
Softmax classification head — *loss-ablation control* for the continual
open-set experiment (vs ProxyAnchor). NOT a COCONUT method component; it exists
only so the paper can show, in the *same* real system, that the training loss is
what determines forgetting (ProxyAnchor retains, classification losses forget).

공정성 설계 (WACV-defensible):
- **Pre-allocated** weight `W[max_classes, D]` → 클래스가 늘어도 텐서 재생성이 없어
  optimizer state가 *구조적으로 보존*된다(ProxyAnchor의 동적 재생성+state 보존
  머신을 복제할 필요 없음 = 같은 대우를 더 단순·안전하게).
- 등록된 클래스 컬럼만 CE에 참여(logits를 active 컬럼으로 slice) → open-set lifelong
  에서 미등록 클래스가 손실에 끼지 않음.
- mode (이름은 표준 용어; 임의 작명 아님):
  - `'cosine_softmax'` = **NormFace** (Wang et al., ACM MM 2017; aka "normalized
    softmax", "cosine softmax" — Wojke & Bewley 2018): `s·cos(norm(feat), norm(W))`
    + CE, **margin 없음**, **feature-mean init** → ProxyAnchor와 *기하·init 동일*,
    손실 형태만 다름(loss-only 귀속). ⚠️ margin이 없으므로 CosFace/ArcFace가 *아님*
    (그들은 cos에 margin을 더함). 논문 인용 시 NormFace로 citing.
  - `'vanilla_softmax'`: 표준 `nn.Linear(D, C)`(weight+bias) + CE = naive-CL 하한.
- head LR은 trainer가 별도 param group(`base_lr * softmax_lr_ratio`)으로 부여 →
  ProxyAnchor proxy와 동등하게 빠르게 학습(베이스라인 불구화 방지).

공정성의 *결과* 검증은 trainer 밖에서: 각 사용자 등록 직후 peak-TAR이 ProxyAnchor와
비슷한지(= 베이스라인이 잘 배웠는지) 확인 → 잘 배웠으면 이후 하락은 진짜 망각.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxHead(nn.Module):
    def __init__(self, embedding_size: int, max_classes: int,
                 mode: str = "cosine_softmax", scale: float = 32.0):
        # scale s = ProxyAnchor alpha(32)와 동일하게 두면 둘이 같은 normalized-cosine 위의
        # 동일 스케일 → proxy vs cosine_softmax가 *손실 집계 형태만* 다른 matched control.
        super().__init__()
        if mode not in ("cosine_softmax", "vanilla_softmax"):
            raise ValueError(f"unknown softmax head mode: {mode!r}")
        self.embedding_size = int(embedding_size)
        self.mode = mode
        self.scale = float(scale)
        # 미리 max_classes만큼 할당 (동적 재생성 없음 = optimizer state 보존).
        # init = ProxyAnchor add_classes의 fallback과 동일(kaiming_normal fan_out) → repo 일관.
        self.W = nn.Parameter(torch.empty(int(max_classes), int(embedding_size)))
        nn.init.kaiming_normal_(self.W, mode="fan_out")
        # vanilla 모드만 bias(표준 nn.Linear와 동일). cosine(NormFace)은 hypersphere라 bias 없음.
        self.b = nn.Parameter(torch.zeros(int(max_classes))) if mode == "vanilla_softmax" else None
        self.class_to_idx = {}   # {user_id: column}
        self.num_classes = 0

    @torch.no_grad()
    def add_classes(self, class_ids, feature_means: dict = None):
        """새 사용자에 컬럼 배정. cosine 모드 + feature_means 주어지면 평균으로 init
        (ProxyAnchor proxy init과 동일). vanilla면 kaiming init 유지."""
        for cid in class_ids:
            cid = int(cid)
            if cid in self.class_to_idx:
                continue
            col = self.num_classes
            if col >= self.W.shape[0]:
                raise RuntimeError(
                    f"SoftmaxHead 용량 초과: max_classes={self.W.shape[0]}, "
                    f"등록 시도 {col+1}. num_experiences를 늘리세요.")
            if (self.mode == "cosine_softmax" and feature_means
                    and cid in feature_means):
                self.W[col] = F.normalize(
                    feature_means[cid].float().to(self.W.device), p=2, dim=0)
            self.class_to_idx[cid] = col
            self.num_classes += 1

    def forward(self, X, T):
        """X:[B,D] 임베딩, T:[B] user_id 레이블 → cross-entropy 스칼라."""
        if self.num_classes == 0:
            return torch.zeros([], device=X.device, dtype=X.dtype)
        Wa = self.W[:self.num_classes]                       # active 컬럼만
        if self.mode == "cosine_softmax":
            logits = self.scale * (F.normalize(X, dim=1) @ F.normalize(Wa, dim=1).t())
        else:
            logits = X @ Wa.t()                              # 표준 linear
            if self.b is not None:
                logits = logits + self.b[:self.num_classes]  # + bias
        labels = T.to(dtype=torch.long, device=X.device)
        # 미등록 레이블은 건너뜀(정상 흐름엔 없음) — 안전 가드
        known = torch.tensor([int(y.item()) in self.class_to_idx for y in labels],
                             device=X.device, dtype=torch.bool)
        if not known.any():
            return torch.zeros([], device=X.device, dtype=X.dtype)
        logits = logits[known]
        tgt = torch.tensor([self.class_to_idx[int(y.item())] for y in labels[known]],
                           device=X.device, dtype=torch.long)
        return F.cross_entropy(logits, tgt)
