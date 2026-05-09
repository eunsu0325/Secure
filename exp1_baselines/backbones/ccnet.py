"""CCNet backbone wrapper (Yang et al. 2023, IEEE TIFS).

Source attribution:
    Original repo:   https://github.com/Zi-YuanYang/CCNet.git
    Reference:       Z. Yang, H. Huangfu, L. Leng, B. Zhang, A. B. J. Teoh,
                     and Y. Zhang, "Comprehensive Competition Mechanism in
                     Palmprint Recognition," IEEE Transactions on
                     Information Forensics and Security, vol. 18,
                     pp. 5160-5170, 2023. DOI: 10.1109/TIFS.2023.3306104.
    Acronym:         CCNet = Comprehensive Competition Network.

Building blocks (`GaborConv2d`, `SELayer`, `CompetitiveBlock_Mul_Ord_Comp`)
are vendored verbatim from the original repository to preserve the
receptive-field design exactly.

CCNet upstream training setup (verified by direct fetch of train.py and
inspection of models/ccnet.py — plan §V16.1):
  - Optimizer: Adam, lr=0.001 (no weight decay, no momentum specified;
    PyTorch defaults betas=(0.9, 0.999), wd=0).
  - LR scheduler: StepLR(step_size=500, gamma=0.8).
  - Defaults: batch_size=1024, epoch_num=3000.
  - Loss: 0.8 * CE + 0.2 * SupCon. CE is computed on `ArcMarginProduct`
    logits (s=30, m=0.5). SupCon is computed on a 6144-D dual-view
    `fe = concat(fc, fc1)` with temperature τ=0.07.
  - Inference embedding (test/eval path): `getFeatureCode(x)` returns
    a **2048-D L2-normalized** feature (the fc1 output L2-normalized).
    This is what the upstream test loop uses for matching.
  - Note: upstream `forward(x)` returns `(logits, F.normalize(6144D fe))`,
    where the 6144-D fe is used inside the SupCon training loss only.
    The 6144-D fe is NOT the upstream inference embedding.

Wrapper deviations from upstream (V17.8 / V16.3 deviation list):
  D1. The built-in `ArcMarginProduct(2048, num_classes, s=30, m=0.5)`
      head is removed. Training uses our shared
      `exp1_baselines/losses/arcface.py:ArcFaceHead` (s=48, m=0.5) so
      that MFN-ArcFace and CCNet-ArcFace share the SAME loss-side recipe
      (recipe parity per plan §IER-9 + §V17.8).
  D2. `forward(x)` returns a single tensor: the **2048-D L2-normalized
      inference embedding** (matching upstream `getFeatureCode`). This
      eliminates the upstream 2048/6144 train/test asymmetry and matches
      the interface of `MobileFaceNet.forward(x)`. The classifier head
      is attached externally during training, identically to MFN-ArcFace.
  D3. The 6144-D `fe` is NOT exposed by this wrapper. It was used in
      upstream as the SupCon training target, but our recipe uses
      ArcFace CE only (no SupCon), so the 6144-D space is not relevant.
  D4. Dropout(p=0.5): upstream applied dropout only to the 2048-D
      h2 head-path, AFTER fe was built. In `getFeatureCode` (the
      inference path), dropout is NOT applied. Our wrapper matches this
      inference-path behavior (no dropout in `forward`). The
      `use_dropout` flag is retained for ablation/sanity (default False
      per recipe parity with MFN-ArcFace which also has no dropout in
      the embedding path).
  D5. Optimizer / scheduler / batch / epochs / loss: trained under our
      standardized ArcFace recipe (SGD lr=0.01 mom=0.9 wd=1e-4, batch
      128, 50 epochs cosine + 1ep warmup, no SupCon). Total iteration
      budget is ~8x fewer than upstream's ~18000 — see V17.10 TPIR
      gate for the L1 50ep / L2 200ep escalation rule.

Input contract:
  - shape: [B, 1, 128, 128] (grayscale; CCNet author native, V17.2)
  - per-pixel value range: [-1, 1] (matches MFN-ArcFace normalization
    convention to keep optimizer dynamics comparable)

Output contract:
  - shape: [B, 4096] L2-normalized along dim=-1
  - cosine similarity between embeddings = standard inner product
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vendored from CCNet repository (verbatim; do not edit)
# ---------------------------------------------------------------------------

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution layer (LGC).

    Vendored from Yang et al. CCNet, models/ccnet.py.
    """

    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # Match upstream explicit requires_grad on all Gabor parameters
        # (gamma/sigma/f learnable; theta/psi fixed). Defaults would be the
        # same but we restate them so reviewers can confirm parity at a glance.
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float())
            * math.pi / channel_out,
            requires_grad=False,
        )
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def gen_gabor_bank(self, kernel_size, channel_in, channel_out,
                       sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin, ymin = -xmax, -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)
        x_theta = (x * torch.cos(theta.view(-1, 1, 1, 1))
                   + y * torch.sin(theta.view(-1, 1, 1, 1)))
        y_theta = (-x * torch.sin(theta.view(-1, 1, 1, 1))
                   + y * torch.cos(theta.view(-1, 1, 1, 1)))
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2)
            / (8 * sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(
            2 * math.pi * f.view(-1, 1, 1, 1) * x_theta
            + psi.view(-1, 1, 1, 1)
        )
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.gen_gabor_bank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi,
        )
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    """Competitive Block (CB = LGC + soft-argmax + PPU)."""

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(
            channel_in=channel_in, channel_out=n_competitor,
            kernel_size=ksize, stride=2, padding=ksize // 2,
            init_ratio=init_ratio,
        )
        self.gabor_conv2d2 = GaborConv2d(
            channel_in=n_competitor, channel_out=n_competitor,
            kernel_size=ksize, stride=2, padding=ksize // 2,
            init_ratio=init_ratio,
        )
        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.conv1_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)
        self.weight_chan = weight
        self.weight_spa = (1 - weight) / 2

    def forward(self, x):
        # 1st-order
        x = self.gabor_conv2d(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)
        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)
        # 2nd-order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)
        xx = torch.cat(
            (x_1.view(x_1.shape[0], -1), x_2.view(x_2.shape[0], -1)),
            dim=1,
        )
        return xx


# ---------------------------------------------------------------------------
# Wrapper for exp1_baselines pipeline (head-stripped, interface-unified)
# ---------------------------------------------------------------------------

class CCNetBackbone(nn.Module):
    """Feature-only CCNet wrapped to match MobileFaceNet's interface.

    - Input  : [B, 1, 128, 128] (grayscale, value range [-1, 1])
    - Output : [B, 2048] L2-normalized along dim=-1

    The output is the **upstream getFeatureCode-style inference embedding**:
    the fc1 output (2048-D) L2-normalized along dim=-1. This matches the
    feature that Yang et al.'s released test loop uses for matching
    (`net.getFeatureCode(data)` in upstream train.py). See plan §V16.2 /
    §V17.8 for the rationale (matches upstream inference path; eliminates
    the upstream 2048/6144 train/test asymmetry).

    Architecture trace at 128x128 input (verified against upstream forward):
      cb_i  produces  spatial[15x15]*16ch + spatial[7x7]*16ch  per competitive
            block when input is 128. All three CBs share this output shape
            because n_competitor differs in channel count of the LGC, not in
            the post-PPU output channel count (o1//2 = 16 in every case).
      flat per CB = 15*15*16 + 7*7*16 = 3600 + 784 = 4384
      total      = 3 * 4384 = 13152  -> matches Linear(13152, 4096)
      fc1(4096)  -> 2048
      output = F.normalize(fc1(fc(z)), dim=-1) -> 2048-D L2-normalized

    The 13152 spatial dim is hardcoded for 128x128 input; using 112x112 input
    would produce 9840 != 13152 and the forward would error. We deliberately
    keep CCNet at its author-specified 128 input rather than padding/resizing
    to a unified 112 (V17.2 lock — architecturally required).

    The upstream 6144-D `fe = concat(fc, fc1)` was used as the SupCon
    training target. We remove SupCon (recipe parity with MFN), so the
    6144-D space is not relevant; we expose only the 2048-D inference
    embedding (V17.8 D2/D3).
    """

    def __init__(self, weight: float = 0.8, use_dropout: bool = False) -> None:
        """
        Args:
            weight: CCNet author's competitive weight (`weight_chan` parameter
                    in CompetitiveBlock); 0.8 matches the released config.
            use_dropout: when True, apply Dropout(p=0.5) to the 2048-D h2
                    BEFORE L2-normalization (matches the upstream training
                    head-path dropout). When False (default), no dropout is
                    applied — matches the upstream `getFeatureCode` inference
                    path and MFN-ArcFace's no-dropout embedding path.
                    Default False is required for V17.8 wrapper definition.
                    The flag is exposed for ablation/sanity only.
        """
        super().__init__()
        self.weight = float(weight)
        self.use_dropout = bool(use_dropout)
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17,
            init_ratio=1, weight=weight,
        )
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8,
            init_ratio=0.5, o2=24, weight=weight,
        )
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3,
            init_ratio=0.25, weight=weight,
        )
        self.fc = nn.Linear(13152, 4096)
        self.fc1 = nn.Linear(4096, 2048)
        # V17.8 D4: dropout is OFF in the inference path (matches upstream
        # getFeatureCode). The use_dropout=True flag enables it during training
        # only, for ablation/sanity. Default False per recipe parity with MFN.
        self.drop = nn.Dropout(p=0.5) if self.use_dropout else nn.Identity()

    @property
    def embedding_dim(self) -> int:
        return 2048

    @staticmethod
    def _assert_embedding_contract(
        embedding: torch.Tensor, expected_dim: int = 2048,
    ) -> None:
        """V17.8 mandatory code-level asserts on the wrapper output."""
        assert embedding.ndim == 2, (
            f"expected 2D embedding tensor (B, D); got shape {tuple(embedding.shape)}"
        )
        assert embedding.shape[1] == expected_dim, (
            f"expected embedding_dim {expected_dim}; got {embedding.shape[1]}. "
            "If this changed, V17.8 lock is broken — verify wrapper edit."
        )
        assert torch.isfinite(embedding).all().item(), (
            "embedding contains NaN or Inf; gradient blow-up or numerical "
            "instability. Investigate before continuing."
        )
        norms = embedding.norm(dim=1)
        unit = torch.ones_like(norms)
        assert torch.allclose(norms, unit, atol=1e-4), (
            f"embedding not L2-normalized; norm mean={norms.mean().item():.4f}, "
            f"std={norms.std().item():.4f}. V17.8 requires unit-norm output."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)
        z = torch.cat((x1, x2, x3), dim=1)
        h1 = self.fc(z)              # 4096 (used internally; not exposed)
        h2 = self.fc1(h1)            # 2048
        if self.training:
            h2 = self.drop(h2)       # nn.Identity() if use_dropout=False
        out = F.normalize(h2, dim=-1)  # 2048D L2-normalized (matches getFeatureCode)
        self._assert_embedding_contract(out, expected_dim=2048)
        return out


def ccnet(weight: float = 0.8, use_dropout: bool = False) -> CCNetBackbone:
    """Factory matching the API style of `iresnet50`."""
    return CCNetBackbone(weight=weight, use_dropout=use_dropout)


if __name__ == "__main__":
    net = ccnet(weight=0.8, use_dropout=False)
    inp = torch.randn(2, 1, 128, 128)
    out = net(inp)
    print(f"input: {inp.shape}  output: {out.shape}  "
          f"embedding_dim={net.embedding_dim}  use_dropout={net.use_dropout}")
    assert out.shape == (2, 2048), f"expected (2, 2048), got {tuple(out.shape)}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), \
        f"output not L2-normalized; norms={norms}"
    print("ccnet wrapper smoke ok")
