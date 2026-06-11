"""
Tests for MRS components (plan §H) — all three are default-OFF and must be
*byte-identical* to current behaviour when off.

  ⓐ SSLConsistencyLoss   (coconut/losses/ssl_consistency.py)
  ⓑ ClassBalancedBuffer.sample(eligible) + CohortScheduler  (coconut/memory/)
  ⓒ DecayPredictor       (coconut/memory/decay_predictor.py)

Run:  python tests/test_replay_scheduler.py      (no pytest needed)
  or: pytest tests/test_replay_scheduler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from coconut.losses import SSLConsistencyLoss
from coconut.memory import ClassBalancedBuffer, CohortScheduler, DecayPredictor
from coconut.evaluation.forgetting_tracker import ForgettingTracker


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_buffer(n_classes=5, per_class=8, max_size=200):
    """ClassBalancedBuffer populated with int 'data' so labels are recoverable."""
    buf = ClassBalancedBuffer(max_size=max_size, adaptive_size=True)
    data, labels = [], []
    for c in range(n_classes):
        for j in range(per_class):
            data.append(c * 1000 + j)  # unique, label-encoding payload
            labels.append(c)
    buf.update_from_dataset(data, labels)
    return buf


# --------------------------------------------------------------------------- #
# ⓐ SSL consistency loss
# --------------------------------------------------------------------------- #
def test_ssl_cosine_identical_views_is_zero():
    loss = SSLConsistencyLoss(mode="cosine")
    x = torch.randn(16, 32)
    out = loss(x, x.clone())
    assert torch.allclose(out, torch.zeros(()), atol=1e-6), out.item()


def test_ssl_cosine_positive_for_disagreeing_views():
    loss = SSLConsistencyLoss(mode="cosine")
    torch.manual_seed(0)
    a = torch.randn(16, 32)
    b = torch.randn(16, 32)
    assert loss(a, b).item() > 0.1


def test_ssl_proto_kl_identical_views_is_zero():
    loss = SSLConsistencyLoss(mode="proto_kl", temperature=0.1)
    torch.manual_seed(1)
    x = torch.randn(16, 32)
    protos = torch.randn(5, 32)
    out = loss(x, x.clone(), prototypes=protos)
    assert torch.allclose(out, torch.zeros(()), atol=1e-6), out.item()


def test_ssl_proto_kl_falls_back_to_cosine_without_prototypes():
    loss = SSLConsistencyLoss(mode="proto_kl")
    x = torch.randn(8, 16)
    # no prototypes -> cosine fallback -> identical views give 0
    out = loss(x, x.clone(), prototypes=None)
    assert torch.allclose(out, torch.zeros(()), atol=1e-6)


def test_ssl_deterministic():
    loss = SSLConsistencyLoss(mode="cosine")
    torch.manual_seed(2)
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    assert loss(a, b).item() == loss(a, b).item()


def test_ssl_shape_mismatch_raises():
    loss = SSLConsistencyLoss()
    try:
        loss(torch.randn(4, 8), torch.randn(4, 9))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


# --------------------------------------------------------------------------- #
# ⓑ buffer.sample byte-identity + eligible filter
# --------------------------------------------------------------------------- #
def test_buffer_sample_none_is_byte_identical():
    """eligible_class_ids=None must consume identical RNG -> identical output."""
    buf = _make_buffer()
    n = 12
    torch.manual_seed(123)
    a = buf.sample(n)
    torch.manual_seed(123)
    b = buf.sample(n, eligible_class_ids=None)
    assert a == b, "None branch diverged from legacy sample()"


def test_buffer_sample_determinism():
    buf = _make_buffer()
    torch.manual_seed(7)
    a = buf.sample(10)
    torch.manual_seed(7)
    b = buf.sample(10)
    assert a == b


def test_buffer_sample_eligible_restricts_classes():
    buf = _make_buffer(n_classes=5, per_class=8)
    torch.manual_seed(0)
    data, labels, _ = buf.sample(12, eligible_class_ids={0, 2})
    assert set(labels).issubset({0, 2}), set(labels)
    assert len(labels) <= 12


def test_buffer_sample_eligible_empty_intersection():
    buf = _make_buffer(n_classes=3)
    out = buf.sample(10, eligible_class_ids={999})
    assert out == ([], [], [])


def test_buffer_sample_budget_preserved():
    buf = _make_buffer(n_classes=4, per_class=20, max_size=400)
    _, labels, _ = buf.sample(16, eligible_class_ids={0, 1, 2, 3})
    # enough data per class -> full budget honoured
    assert len(labels) == 16


# --------------------------------------------------------------------------- #
# ⓑ cohort scheduler
# --------------------------------------------------------------------------- #
def test_cohort_off_when_interval_one():
    sched = CohortScheduler(recall_interval=1)
    sched.register(0); sched.register(1)
    assert sched.enabled is False
    assert sched.eligible(0) is None
    assert sched.eligible(5) is None


def test_cohort_off_during_warmup():
    sched = CohortScheduler(recall_interval=2, warmup_classes=5)
    for c in range(3):
        sched.register(c)
    assert sched.eligible(0) is None  # < warmup -> off
    for c in range(3, 6):
        sched.register(c)
    assert sched.eligible(0) is not None  # >= warmup -> on


def test_cohort_phases_partition_all_classes():
    R = 2
    sched = CohortScheduler(recall_interval=R)
    for c in range(4):
        sched.register(c)
    union = set()
    for step in range(R):
        union |= sched.eligible(step)
    assert union == {0, 1, 2, 3}  # R steps cover everyone exactly once


def test_cohort_register_dedup_preserves_order():
    sched = CohortScheduler(recall_interval=3)
    sched.register(10); sched.register(10); sched.register(20)
    assert sched._order == [10, 20]


# --------------------------------------------------------------------------- #
# ⓒ decay predictor
# --------------------------------------------------------------------------- #
def _tracker_with(trajectories, metric="1-eer"):
    t = ForgettingTracker(primary_metric=metric)
    for uid, vals in trajectories.items():
        for step, v in enumerate(vals):
            t.update_user_performance(uid, {metric: v}, test_step=step, experience_id=step)
    return t


def test_decay_off_when_disabled():
    t = _tracker_with({100: [0.9, 0.7, 0.4]})
    pred = DecayPredictor(enabled=False, floor=0.5)
    assert pred.at_risk(t) == set()


def test_decay_flags_declining_below_floor():
    t = _tracker_with({100: [0.9, 0.7, 0.55]})
    pred = DecayPredictor(enabled=True, metric="1-eer", floor=0.5,
                          horizon=1, min_history=2, min_baseline=0.0)
    assert 100 in pred.at_risk(t)


def test_decay_new_user_not_flagged():
    t = _tracker_with({101: [0.9]})  # single eval < min_history
    pred = DecayPredictor(enabled=True, floor=0.95, min_history=2)
    assert 101 not in pred.at_risk(t)


def test_decay_low_baseline_filtered():
    t = _tracker_with({102: [0.2, 0.15, 0.1]})
    pred = DecayPredictor(enabled=True, floor=0.5, min_baseline=0.5)
    assert 102 not in pred.at_risk(t)


def test_decay_improving_not_flagged():
    t = _tracker_with({103: [0.4, 0.6, 0.8]})
    pred = DecayPredictor(enabled=True, floor=0.9)
    assert 103 not in pred.at_risk(t)


def test_decay_cap_limits_count():
    t = _tracker_with({
        1: [0.9, 0.6, 0.4],   # steep decline -> most at risk
        2: [0.9, 0.8, 0.7],   # mild decline
    })
    pred = DecayPredictor(enabled=True, floor=0.95, horizon=1, cap=1)
    risk = pred.at_risk(t)
    assert len(risk) == 1
    assert risk == {1}  # the steeper decliner wins the single slot


def test_decay_floor_is_caller_controlled():
    """floor must be injected (raw-cosine), not hardcoded/derived from tau_cos."""
    t = _tracker_with({1: [0.9, 0.7, 0.6]})
    lo = DecayPredictor(enabled=True, floor=0.3, horizon=1).at_risk(t)
    hi = DecayPredictor(enabled=True, floor=0.9, horizon=1).at_risk(t)
    assert lo == set()       # predicted ~0.5 > 0.3 -> safe
    assert hi == {1}         # predicted ~0.5 < 0.9 -> at risk


def test_decay_lower_is_better_eer():
    """metric='eer' (lower better): rising toward/over floor -> at risk."""
    t = _tracker_with({1: [0.02, 0.05, 0.09]}, metric="eer")
    # rising eer, floor=0.12 (max acceptable eer). predicted ~0.13 > 0.12 -> risk
    pred = DecayPredictor(enabled=True, metric="eer", higher_is_better=False,
                          floor=0.12, horizon=1)
    assert 1 in pred.at_risk(t)
    # same trajectory but treated as higher-is-better -> slope<0 path -> not flagged
    wrong = DecayPredictor(enabled=True, metric="eer", higher_is_better=True,
                           floor=0.12, horizon=1)
    assert wrong.at_risk(t) == set()


def test_decay_lower_is_better_improving_not_flagged():
    """eer falling (improving) must NOT be flagged."""
    t = _tracker_with({1: [0.09, 0.05, 0.02]}, metric="eer")
    pred = DecayPredictor(enabled=True, metric="eer", higher_is_better=False,
                          floor=0.12, horizon=1)
    assert pred.at_risk(t) == set()


def test_decay_min_baseline_none_disables_filter():
    """default min_baseline=None -> never filters on baseline."""
    t = _tracker_with({1: [0.9, 0.7, 0.55]})
    pred = DecayPredictor(enabled=True, floor=0.5, horizon=1)  # min_baseline default None
    assert 1 in pred.at_risk(t)


def test_decay_ema_mode_runs():
    t = _tracker_with({1: [0.9, 0.7, 0.5, 0.4]})
    pred = DecayPredictor(enabled=True, floor=0.45, slope_mode="ema",
                          ema_alpha=0.5, horizon=1)
    # declining EMA -> predicted below 0.45
    assert 1 in pred.at_risk(t)


# --------------------------------------------------------------------------- #
# runner
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
