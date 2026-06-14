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
# C2 — one-to-many KL (VBM Eq3) — plan §L C2 / D6 / D7
# --------------------------------------------------------------------------- #
def test_c2_zero_for_identical_views():
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    protos = torch.randn(5, 8)
    feat = torch.randn(4, 8)
    kl = ssl.one_to_many(feat, [feat.clone(), feat.clone()], prototypes=protos)
    assert kl.item() < 1e-5, kl.item()


def test_c2_positive_for_different_views():
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    torch.manual_seed(0)
    kl = ssl.one_to_many(torch.randn(4, 8), [torch.randn(4, 8)], prototypes=torch.randn(5, 8))
    assert kl.item() > 0


def test_c2_weak_anchor_is_detached():
    """D6: weak(anchor)엔 grad가 안 흐르고 strong에만 흘러야 함."""
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    protos = torch.randn(5, 8)
    weak = torch.randn(4, 8, requires_grad=True)
    strong = torch.randn(4, 8, requires_grad=True)
    ssl.one_to_many(weak, [strong], prototypes=protos).backward()
    assert weak.grad is None or weak.grad.abs().sum().item() == 0.0, "weak must be detached (D6)"
    assert strong.grad is not None and strong.grad.abs().sum().item() > 0, "strong must receive grad"


def test_c2_cosine_fallback_without_prototypes():
    """D7: prototypes=None → cosine 폴백, 동일 뷰면 0."""
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    feat = torch.randn(4, 8)
    kl = ssl.one_to_many(feat, [feat.clone()], prototypes=None)
    assert kl.item() < 1e-5, kl.item()


def test_c2_averages_over_multiple_strong_views():
    """V=3 (strong 2개) 동작 + 유한·비음수."""
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    torch.manual_seed(0)
    kl = ssl.one_to_many(torch.randn(4, 8), [torch.randn(4, 8), torch.randn(4, 8)],
                         prototypes=torch.randn(5, 8))
    assert torch.isfinite(kl).item() and kl.item() >= 0


def test_c2_empty_strong_is_zero():
    from coconut.losses import SSLConsistencyLoss
    ssl = SSLConsistencyLoss(mode='proto_kl')
    kl = ssl.one_to_many(torch.randn(4, 8), [], prototypes=torch.randn(5, 8))
    assert kl.item() == 0.0


# --------------------------------------------------------------------------- #
# C3 — activation-state scheduler (_aavb_select_eligible) — plan §L C3 / D15
# --------------------------------------------------------------------------- #
def _fake_aavb(hist, floor, groups, W=10, spu=5):
    """trainer 전체 구성 없이 _aavb_select_eligible 로직만 테스트하는 fake self."""
    from coconut.training.trainer import COCONUTTrainer
    from coconut.memory import DecayPredictor
    class _F: pass
    f = _F()
    f._qar_raw_floor = floor
    f._qar_score_history = hist
    f.aavb_peak_window = W
    f.aavb_samples_per_user_target = spu
    f.decay_predictor = DecayPredictor(enabled=True, higher_is_better=True,
                                       floor=floor or 0.0, horizon=1, min_history=2)
    class _B: pass
    b = _B(); b.buffer_groups = {g: None for g in groups}
    f.memory_buffer = b
    return f, COCONUTTrainer._aavb_select_eligible


def test_c3_warmup_returns_none():
    """floor/history 없으면 None(균등 폴백)."""
    f, m = _fake_aavb({}, None, [1, 2, 3])
    assert m(f, 10) is None


def test_c3_decline_first_and_chronic_excluded():
    """at-risk 강제 + decline 큰 순 선택 + 만성약자(decline≈0) 예산압박 시 제외."""
    hist = {1: [0.9, 0.6, 0.4],    # at-risk (declining below floor 0.5)
            2: [0.9, 0.8, 0.7],    # decline 0.2 (above floor)
            3: [0.8, 0.78, 0.76],  # decline 0.04
            4: [0.3, 0.3, 0.3]}    # chronic flat (decline 0, slope 0)
    f, m = _fake_aavb(hist, 0.5, [1, 2, 3, 4], spu=5)
    elig = m(f, 15)               # K = max(|at_risk|=1, 15//5=3) = 3
    assert elig is not None
    assert 1 in elig, "at-risk must be forced"
    assert 2 in elig, "biggest decline must be selected"
    assert 4 not in elig, "chronic-weak (decline~0) excluded under budget"


def test_c3_new_user_protected():
    """history < min_history(2) 사용자는 보호(포함)."""
    hist = {1: [0.9, 0.8, 0.7], 2: [0.6]}   # 2 = 1 eval → new
    f, m = _fake_aavb(hist, 0.5, [1, 2], spu=5)
    elig = m(f, 5)
    assert elig is not None and 2 in elig, "new user must be protected"


def test_c3_all_eligible_returns_none():
    """전원이 eligible이면(=균등과 동등) None."""
    hist = {1: [0.9, 0.6, 0.3], 2: [0.8, 0.5, 0.2]}  # 둘 다 floor 아래로 declining → at_risk
    f, m = _fake_aavb(hist, 0.5, [1, 2], spu=5)
    assert m(f, 5) is None


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
