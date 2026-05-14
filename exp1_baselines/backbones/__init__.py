"""Backbones for exp1_baselines (vendored from InsightFace + CCNet, with attribution).

V19 Path 3b adds `ccm_mfn`: a CCNet-inspired competitive-mechanism residual
adapter integrated into MobileFaceNet's early feature stage. Not a
reproduction of CCNet or RegPalm — see exp1_baselines/backbones/ccm_mfn.py
docstring and plan §V19 for the inspired-variant framing.
"""

from exp1_baselines.backbones.mobilefacenet import MobileFaceNet
from exp1_baselines.backbones.iresnet import iresnet50
from exp1_baselines.backbones.ccnet import ccnet
from exp1_baselines.backbones.ccm_mfn import ccm_mfn

__all__ = ["MobileFaceNet", "iresnet50", "ccnet", "ccm_mfn"]
