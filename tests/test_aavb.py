"""
Tests for AAVB Phase 1 (view-batch / asymmetric V-view) — plan §L C1.

Default OFF (use_aavb_views=False) must be byte-identical to the existing
dual-view path; AAVB ON returns V views = 1 weak(anchor) + (V-1) strong.

Run:  python tests/test_aavb.py      (no pytest needed)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from coconut.data.datasets import MemoryDataset, BaseVeinDataset
from coconut.data.transforms import get_aavb_weak_transform, get_aavb_strong_transform


# --------------------------------------------------------------------------- #
# helpers — marker transforms so the chosen view is identifiable
# --------------------------------------------------------------------------- #
def _tmp_img(val=200):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.png")
    Image.new("L", (16, 16), color=val).save(p)
    return p


WEAK = lambda im: torch.zeros(1, 4, 4)    # marker for "weak" view
STRONG = lambda im: torch.ones(1, 4, 4)   # marker for "strong" view


# --------------------------------------------------------------------------- #
# byte-identical default (use_aavb_views=False)
# --------------------------------------------------------------------------- #
def test_dual_view_default_returns_two():
    """use_aavb_views=False(기본) → dual-view 2개 (기존 경로 유지)."""
    ds = MemoryDataset(paths=[_tmp_img()], labels=[0], transform=WEAK, train=True)
    data, label = ds[0]
    assert isinstance(data, list) and len(data) == 2, f"expected 2 views, got {data}"
    assert label == 0


def test_base_single_view_unchanged():
    """train=False(dual_views=False) → 단일 뷰 (기존 경로)."""
    ds = BaseVeinDataset(paths=[_tmp_img()], labels=[0], transform=WEAK, train=False)
    data, _ = ds[0]
    assert not isinstance(data, list), "single view should not be a list"


# --------------------------------------------------------------------------- #
# AAVB asymmetric V-view
# --------------------------------------------------------------------------- #
def test_aavb_returns_n_views():
    ds = MemoryDataset(paths=[_tmp_img()], labels=[0], transform=None, train=True,
                       use_aavb_views=True, n_views=3,
                       weak_transform=WEAK, strong_transform=STRONG)
    data, _ = ds[0]
    assert len(data) == 3, f"expected 3 views, got {len(data)}"


def test_aavb_view0_weak_rest_strong():
    """뷰0=weak(anchor), 뷰1..=strong (비대칭, plan §L C1/D2)."""
    ds = MemoryDataset(paths=[_tmp_img()], labels=[0], transform=None, train=True,
                       use_aavb_views=True, n_views=3,
                       weak_transform=WEAK, strong_transform=STRONG)
    data, _ = ds[0]
    assert torch.equal(data[0], torch.zeros(1, 4, 4)), "view0 must be weak"
    assert torch.equal(data[1], torch.ones(1, 4, 4)), "view1 must be strong"
    assert torch.equal(data[2], torch.ones(1, 4, 4)), "view2 must be strong"


def test_aavb_v2_one_weak_one_strong():
    """V=2 AAVB = weak+strong 1개씩 (개수는 dual과 같으나 비대칭)."""
    ds = MemoryDataset(paths=[_tmp_img()], labels=[0], transform=None, train=True,
                       use_aavb_views=True, n_views=2,
                       weak_transform=WEAK, strong_transform=STRONG)
    data, _ = ds[0]
    assert len(data) == 2
    assert torch.equal(data[0], torch.zeros(1, 4, 4))
    assert torch.equal(data[1], torch.ones(1, 4, 4))


# --------------------------------------------------------------------------- #
# real transforms produce valid tensors
# --------------------------------------------------------------------------- #
def test_aavb_real_transforms_shape():
    """get_aavb_weak/strong이 [1,H,W] 텐서를 만든다 (실제 증강 파이프라인)."""
    img = Image.new("L", (16, 16), color=200)
    weak = get_aavb_weak_transform(imside=16, channels=1)
    strong = get_aavb_strong_transform(imside=16, channels=1)
    w, s = weak(img), strong(img)
    assert w.dim() == 3 and w.shape[0] == 1, w.shape
    assert s.dim() == 3 and s.shape[0] == 1, s.shape


def test_aavb_weak_is_deterministic():
    """weak(anchor)는 결정적(증강 없음) → 같은 입력 두 번 = 동일."""
    img = Image.new("L", (16, 16), color=120)
    weak = get_aavb_weak_transform(imside=16, channels=1)
    a, b = weak(img), weak(img)
    assert torch.equal(a, b), "weak transform must be deterministic (no aug)"


# --------------------------------------------------------------------------- #
# margin = d-prime fix (scale-invariant, bounded) — plan §L 측정 정직성
# --------------------------------------------------------------------------- #
def test_margin_dprime_sign_and_forgetting():
    """분리 좋으면 d'↑, genuine 하락(망각)하면 d'↓."""
    from coconut.evaluation.biometric_metrics import calculate_similarity_margin as m
    imp = np.array([0.2, 0.1, 0.3, 0.25])
    good = m(np.array([0.8, 0.7, 0.75]), imp)
    forgot = m(np.array([0.4, 0.3, 0.35]), imp)
    assert good > 0, good
    assert forgot < good, f"forgetting must lower d' ({forgot} !< {good})"


def test_margin_dprime_scale_invariant():
    """raw cosine든 z-score든(×100) 동일 = 스케일 불변."""
    from coconut.evaluation.biometric_metrics import calculate_similarity_margin as m
    g = np.array([0.8, 0.7, 0.75]); i = np.array([0.2, 0.1, 0.3])
    # 1e-6 floor(div-0 방지)만큼만 차이 → 무시 수준(상대오차 <0.1%)
    assert abs(m(g, i) - m(g * 100, i * 100)) < 1e-2


def test_margin_dprime_bounded_on_zscore_and_degenerate():
    """z-score 대형 입력·degenerate(genuine 1개)서 폭주 없음(|d'|<100)."""
    from coconut.evaluation.biometric_metrics import calculate_similarity_margin as m
    z = m(np.array([12.0, 10.0, 11.0]), np.array([0.5, -0.5, 1.0, 0.0, 2.0]))
    deg = m(np.array([0.9]), np.array([0.2, 0.1, 0.3, 0.25]))
    assert abs(z) < 100 and abs(deg) < 100, (z, deg)


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
