"""Multi-seed Tongji run: order_seed × enroll_seed = 30 trajectories.

Reuses the existing checkpoint + NPZ embeddings from `tongji_full/` and only
re-runs the orchestrator with different (order_seed, enroll_seed). The
score matrix changes per enroll_seed (different K-image selection feeds
different prototypes); order_seed only affects the future enrollment
sequence (slicing). Per IER-7 + plan Seeds section.

This script does NOT retrain the backbone or re-extract embeddings —
30 orchestrator runs total ~5-10 minutes.

Usage:
    python experiments/run_multiseed_tongji.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.eval.orchestrator import run_orchestrator


SOURCE = PROJECT_ROOT / "experiments" / "generated" / "tongji_full"
EMB    = SOURCE / "mfn_tongji_112_embeddings.npz"
MAN    = SOURCE / "manifest.csv"
OUT    = PROJECT_ROOT / "experiments" / "generated" / "tongji_multiseed"

ORDER_SEEDS  = list(range(10))   # 0..9
ENROLL_SEEDS = list(range(3))    # 0..2

TARGET_FPIR = 0.05
T_STEP = 10
K = 3


def main() -> int:
    if not EMB.exists() or not MAN.exists():
        print(f"missing source files in {SOURCE}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    total = len(ORDER_SEEDS) * len(ENROLL_SEEDS)
    done = 0
    start = time.time()
    failures = []

    for o in ORDER_SEEDS:
        for e in ENROLL_SEEDS:
            sub = OUT / f"order{o}_enroll{e}"
            sub.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                meta = run_orchestrator(
                    embeddings_npz=EMB,
                    manifest_csv=MAN,
                    out_dir=sub / "protocols",
                    target_fpir=TARGET_FPIR,
                    t_step=T_STEP,
                    order_seed=o,
                    enroll_seed=e,
                    K=K,
                    enroll_session_id="session1",
                    query_session_id="session2",
                    sessioned_K_pool=10,
                )
                ok = meta["b_vs_c_recal_endpoint_sanity"] == "ok"
                if not ok:
                    failures.append((o, e, meta.get("b_vs_c_recal_endpoint_sanity_error")))
            except Exception as ex:
                failures.append((o, e, str(ex)))
                ok = False
            done += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - start
            eta = (total_elapsed / done) * (total - done)
            print(
                f"[{done:>2}/{total}] order={o} enroll={e} "
                f"sanity={'OK' if ok else 'FAIL'} "
                f"({elapsed:.1f}s, ETA {eta:.0f}s)"
            )

    print(f"\ndone in {time.time()-start:.1f}s; {len(failures)} failure(s)")
    for f in failures:
        print(f"  FAIL: order={f[0]} enroll={f[1]} -> {f[2]}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
