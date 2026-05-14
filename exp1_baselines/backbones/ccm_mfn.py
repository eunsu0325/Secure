"""CCM-on-MFN backbone: MobileFaceNet augmented with a CCNet-inspired
competitive-mechanism residual adapter.

Plan reference: §V19 Path 3b. CCM-on-MFN is **NOT a reproduction** of CCNet
(Yang et al., IEEE TIFS 2023) or RegPalm (Zhong et al., IEEE TIFS 2025). It
is an architecturally inspired variant that integrates CCNet's learnable
Gabor filtering and channel/spatial competition mechanisms into the early
feature stage of MobileFaceNet, packaged as a shape-preserving residual
adapter so it can be used as a drop-in feature-extractor block under our
unified Family-A training recipe (SGD + ArcFace, no SupCon).

Forbidden claims (V19.5): this module must never be described as
"reproducing CCNet" or "reproducing RegPalm" or "equivalent to full CCNet"
or "superior to MFN" or "the SOTA palmprint backbone".

Architecture:
    Input:  [B, 3, 112, 112]
    -> conv1 (3 -> 64, k=3, stride=2, pad=1)             # [B, 64, 56, 56]
    -> x = x + alpha * CCM_Adapter(x)                    # residual, alpha=0.05 init
    -> conv2_dw -> conv_23 -> conv_3 -> conv_34 ->        # rest of MFN, unchanged
       conv_4 -> conv_45 -> conv_5 -> conv_6_sep
    -> gdc                                                # [B, 512]
    Output: [B, 512] (raw, not L2-normalized; ArcFaceHead normalizes)

Why alpha_init=0.05 (V19.3.3a):
    For y = x + alpha * CCM(x), the gradient w.r.t. CCM internal parameters
    is `alpha * (dL/dy) * (dCCM/dtheta)`. With alpha=0 exactly, theta_CCM
    receives zero gradient at step 0 (ReZero's self-warm-up still works,
    but is delayed). With alpha=0.05, both alpha AND theta_CCM receive
    non-zero gradient from step 0, while the forward perturbation from
    MFN is only ~5% (CCM_Adapter output is unit-variance via BN).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from exp1_baselines.backbones.ccnet import GaborConv2d
from exp1_baselines.backbones.mobilefacenet import (
    ConvBlock,
    DepthWise,
    Residual,
    GDC,
)


class CCM_Adapter(nn.Module):
    """CCNet-inspired competitive-mechanism block, shape-preserving.

    Forward: [B, C, H, W] -> [B, C, H, W]

    Pipeline (per V19.3.2):
        x
        -> GaborConv2d (C -> n_competitor, k=ksize, stride=1, pad=ksize//2)
        -> softmax_c + softmax_x + softmax_y, weighted combine
              z1 = w_chan*softmax_c(z) + w_spa*(softmax_x(z) + softmax_y(z))
        -> GaborConv2d (n_competitor -> n_competitor)
        -> softmax_c + softmax_x + softmax_y, weighted combine
              z2 = ...
        -> Conv2d 1x1 (n_competitor -> C)
        -> BatchNorm2d (C)

    This is NOT a port of CCNet's CompetitiveBlock_Mul_Ord_Comp; that block
    flattens the output and applies 4x spatial downsampling, which is
    incompatible with use as a residual adapter inside MFN. Instead, we
    extract three core ideas:
      - Learnable Gabor filtering across multiple orientations (CCNet author's
        n_competitor=9 channels = 9 evenly-spaced theta in [0, pi]).
      - Channel/spatial softmax competition.
      - Author's weighted combination (weight_chan=0.8, weight_spa=0.1).
    and re-package them in a shape-preserving form.
    """

    def __init__(
        self,
        channels: int = 64,
        n_competitor: int = 9,
        ksize: int = 7,
        init_ratio: float = 1.0,
        weight: float = 0.8,
    ) -> None:
        super().__init__()
        if ksize % 2 == 0:
            raise ValueError(
                f"ksize must be odd to preserve spatial shape with padding=ksize//2; got {ksize}"
            )
        self.channels = int(channels)
        self.n_competitor = int(n_competitor)
        self.ksize = int(ksize)
        self.init_ratio = float(init_ratio)
        self.weight_chan = float(weight)
        self.weight_spa = (1.0 - float(weight)) / 2.0

        pad = ksize // 2
        self.gabor1 = GaborConv2d(
            channel_in=channels, channel_out=n_competitor,
            kernel_size=ksize, stride=1, padding=pad,
            init_ratio=init_ratio,
        )
        self.gabor2 = GaborConv2d(
            channel_in=n_competitor, channel_out=n_competitor,
            kernel_size=ksize, stride=1, padding=pad,
            init_ratio=init_ratio,
        )
        self.softmax_c = nn.Softmax(dim=1)
        self.softmax_x = nn.Softmax(dim=2)
        self.softmax_y = nn.Softmax(dim=3)
        # 1x1 projection back to MFN channel count for residual compatibility.
        self.proj_out = nn.Conv2d(n_competitor, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1st-order Gabor + competition
        z = self.gabor1(x)
        z1 = (
            self.weight_chan * self.softmax_c(z)
            + self.weight_spa * (self.softmax_x(z) + self.softmax_y(z))
        )
        # 2nd-order Gabor + competition
        z = self.gabor2(z1)
        z2 = (
            self.weight_chan * self.softmax_c(z)
            + self.weight_spa * (self.softmax_x(z) + self.softmax_y(z))
        )
        out = self.proj_out(z2)
        out = self.bn(out)
        return out


class CCM_MFN(nn.Module):
    """MobileFaceNet + CCM residual adapter at the early feature stage.

    Plan reference: §V19. Layout is identical to InsightFace MobileFaceNet
    except for the addition of (a) a CCM_Adapter applied between conv1 and
    conv2_dw, and (b) a learnable scalar `alpha` that gates the CCM residual:

        x = conv1(x)                          # [B, 64, 56, 56]
        x = x + alpha * CCM_Adapter(x)        # alpha is nn.Parameter (init 0.05)
        x = conv2_dw(x); ...                  # rest of MFN unchanged

    Output: [B, embedding_dim] (default 512), raw (not L2-normalized).
    The shared ArcFaceHead normalizes features during training; the extract
    pipeline applies F.normalize on top, so both training and evaluation see
    L2-normalized embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        fp16: bool = False,
        ccm_n_competitor: int = 9,
        ccm_ksize: int = 7,
        ccm_init_ratio: float = 1.0,
        ccm_weight: float = 0.8,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.fp16 = fp16

        # ---- MFN modules (instantiated individually so we can splice CCM in) ----
        self.conv1 = ConvBlock(3, 64, kernel=3, stride=2, padding=1)
        self.ccm = CCM_Adapter(
            channels=64,
            n_competitor=ccm_n_competitor,
            ksize=ccm_ksize,
            init_ratio=ccm_init_ratio,
            weight=ccm_weight,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.conv2_dw = ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=3, stride=2, padding=1, groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=3, stride=1, padding=1)
        self.conv_34 = DepthWise(64, 128, kernel=3, stride=2, padding=1, groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=3, stride=1, padding=1)
        self.conv_45 = DepthWise(128, 128, kernel=3, stride=2, padding=1, groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=3, stride=1, padding=1)
        self.conv_6_sep = ConvBlock(128, 512, kernel=1, stride=1, padding=0)
        self.gdc = GDC(embedding_dim)

        self._initialize_weights()

        # Re-initialize alpha AFTER _initialize_weights so kaiming init does
        # not overwrite the prescribed init value. nn.Parameter is plain
        # tensor for kaiming, but explicit re-set is safe and explicit.
        with torch.no_grad():
            self.alpha.fill_(float(alpha_init))

    def _initialize_weights(self) -> None:
        """Same as MobileFaceNet._initialize_weights (kaiming for conv/linear,
        unit/zero for BN). Gabor parameters use GaborConv2d's own init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def embedding_dim(self) -> int:
        return self.gdc.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            # V19.3.3: alpha-gated CCM residual at early feature stage [B, 64, 56, 56]
            x = x + self.alpha * self.ccm(x)
            x = self.conv2_dw(x)
            x = self.conv_23(x)
            x = self.conv_3(x)
            x = self.conv_34(x)
            x = self.conv_4(x)
            x = self.conv_45(x)
            x = self.conv_5(x)
            x = self.conv_6_sep(x)
        x = self.gdc(x.float() if self.fp16 else x)
        return x


def ccm_mfn(
    embedding_dim: int = 512,
    fp16: bool = False,
    ccm_n_competitor: int = 9,
    ccm_ksize: int = 7,
    ccm_init_ratio: float = 1.0,
    ccm_weight: float = 0.8,
    alpha_init: float = 0.05,
) -> CCM_MFN:
    """Factory matching the API style of `iresnet50` and `ccnet`."""
    return CCM_MFN(
        embedding_dim=embedding_dim,
        fp16=fp16,
        ccm_n_competitor=ccm_n_competitor,
        ccm_ksize=ccm_ksize,
        ccm_init_ratio=ccm_init_ratio,
        ccm_weight=ccm_weight,
        alpha_init=alpha_init,
    )


if __name__ == "__main__":
    # V19.9 step 2 + step 3 (partial): synthetic forward shape + gradient flow.
    # batch_size=8 minimum to keep BatchNorm1d in GDC stable (batch=2 causes
    # near-zero variance in the embedding dim and rounds gradients to zero
    # under :.6f printing). Real training uses batch=128 (P=32, K=4).
    net = ccm_mfn(embedding_dim=512, alpha_init=0.05)
    net.eval()
    inp = torch.randn(8, 3, 112, 112)
    out = net(inp)
    assert out.shape == (8, 512), f"expected (8, 512), got {tuple(out.shape)}"
    print(f"[ccm_mfn smoke] input {tuple(inp.shape)} -> output {tuple(out.shape)}")
    print(f"  alpha init value: {net.alpha.item():.4f}")
    print(f"  embedding_dim: {net.embedding_dim}")
    assert torch.isfinite(out).all(), "ccm_mfn produced non-finite output"
    norms = out.norm(dim=1)
    print(f"  raw embedding L2-norm range: {norms.min().item():.4f} ~ {norms.max().item():.4f}")
    # Gradient flow check (train mode, dummy loss=sum(out))
    net.train()
    out = net(inp)
    loss = out.sum()
    loss.backward()
    alpha_grad = net.alpha.grad
    gabor_f_grad = net.ccm.gabor1.f.grad
    gabor_sigma_grad = net.ccm.gabor1.sigma.grad
    proj_out_grad = net.ccm.proj_out.weight.grad
    print(f"  alpha.grad: {alpha_grad.item():.3e} (non-zero ⇒ alpha learns)")
    print(f"  ccm.gabor1.f.grad: {gabor_f_grad.item():.3e}")
    print(f"  ccm.gabor1.sigma.grad: {gabor_sigma_grad.item():.3e}")
    print(f"  ccm.proj_out.weight.grad.abs.mean: {proj_out_grad.abs().mean().item():.3e}")
    # All gradients must be strictly non-zero at alpha_init=0.05
    assert alpha_grad is not None and alpha_grad.abs().item() > 0, "alpha gradient is zero"
    assert gabor_f_grad is not None and gabor_f_grad.abs().item() > 0, (
        "CCM internal Gabor f gradient is zero — gradient flow blocked. "
        "Verify alpha_init > 0."
    )
    assert proj_out_grad is not None and proj_out_grad.abs().sum().item() > 0, (
        "CCM proj_out gradient is zero — gradient flow blocked."
    )
    print("ccm_mfn synthetic + gradient sanity OK")
