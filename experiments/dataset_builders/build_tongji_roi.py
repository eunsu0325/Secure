"""Build the Exp1 manifest for Tongji ROI — RepViT-only / leakage-aware.

Input layout (from ``--root``)::

    Tongji/ROI/
        session1/
            00001.bmp 00002.bmp ... 06000.bmp
        session2/
            00001.bmp 00002.bmp ... 06000.bmp

Each filename is a 5-digit image number 00001..06000. ``palm_id = (n-1) // 10``
since each palm-side has 10 images per session, sequential. session1 and
session2 share filenames for the same palm-side (cross-session pairs).

Sample split per palm-side (sorted by numeric image_number):
    enroll = session1[:5]
    dev    = session1[5:10]
    test   = session2[:10]

Recommended split: 100 / 300 / 200 palm-sides.
``future_batch_size = 50``, ``static_gallery_sizes = [100, 200, 300, 400]``.

CCNet-on-Tongji leakage: this builder writes ONLY the RepViT YAML. The
factory's runtime guard (``model_factory._check_tongji_leakage``) is the
second line of defence if a user manually crafts a CCNet config.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_common import (  # noqa: E402
    is_image_file,
    write_all_txt,
    write_identity_split,
    write_sample_split,
    write_metadata,
    write_configs,
    validate_manifest,
)


_FILENAME_RE = re.compile(r"^(?P<n>\d+)$")


def _palm_id_from_n(n: int) -> int:
    return (n - 1) // 10


def _walk_session(session_dir: Path) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = defaultdict(list)
    if not session_dir.is_dir():
        return out
    for p in sorted(session_dir.rglob("*")):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        m = _FILENAME_RE.match(p.stem)
        if not m:
            continue
        n = int(m.group("n"))
        if n < 1:
            continue
        palm_id = _palm_id_from_n(n)
        out[palm_id].append({"path": p, "n": n})
    return out


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Tongji ROI manifest for Exp1 (RepViT-only, appendix)"
    )
    parser.add_argument("--root", required=True, type=str,
                        help="Path to Tongji/ROI containing session1/ and session2/")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_palmsides", default=100, type=int)
    parser.add_argument("--n_future_palmsides", default=300, type=int)
    parser.add_argument("--n_external_palmsides", default=200, type=int)
    parser.add_argument("--future_batch_size", default=50, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = _walk_session(root / "session1")
    s2 = _walk_session(root / "session2")
    if not s1 or not s2:
        raise FileNotFoundError(f"Tongji root must contain session1/ and session2/: {root}")

    excluded_items: List[Dict] = [
        {"reason": "ccnet_tongji_leakage_avoided",
         "note": "Tongji checkpoint = CCNet pretraining set. Only RepViT YAML is emitted."}
    ]

    eligible_palm_ids: List[int] = []
    palm_records: Dict[int, Dict[str, List[Dict]]] = {}
    all_palm_ids = sorted(set(s1) | set(s2))
    for pid in all_palm_ids:
        s1_recs = sorted(s1.get(pid, []), key=lambda r: r["n"])
        s2_recs = sorted(s2.get(pid, []), key=lambda r: r["n"])
        if len(s1_recs) >= 10 and len(s2_recs) >= 10:
            palm_records[pid] = {"s1": s1_recs, "s2": s2_recs}
            eligible_palm_ids.append(pid)
        else:
            excluded_items.append({
                "reason": "incomplete_palm",
                "palm_id": pid, "n_s1": len(s1_recs), "n_s2": len(s2_recs),
            })

    n_base, n_future, n_external = (
        args.n_base_palmsides, args.n_future_palmsides, args.n_external_palmsides
    )
    total = n_base + n_future + n_external
    if len(eligible_palm_ids) < total:
        raise RuntimeError(
            f"Need {total} eligible palm-sides but only {len(eligible_palm_ids)} are complete."
        )

    rng = np.random.RandomState(args.seed)
    shuffled = list(rng.permutation(eligible_palm_ids))
    selected = shuffled[:total]
    base_palms = sorted(int(x) for x in selected[:n_base])
    future_palms = [int(x) for x in selected[n_base:n_base + n_future]]
    external_palms = sorted(int(x) for x in selected[n_base + n_future:total])

    canonical_palm_order = list(base_palms) + list(future_palms) + list(external_palms)
    base_set = set(base_palms)
    future_set = set(future_palms)

    label_to_subject_side: Dict[int, Dict[str, str]] = {}
    sample_split: Dict[int, Dict[str, List[str]]] = {}
    base_ids: List[int] = []
    future_ids: List[int] = []
    external_ids: List[int] = []

    for label, pid in enumerate(canonical_palm_order):
        recs = palm_records[pid]
        s1_paths = [_rel_to_project(r["path"]) for r in recs["s1"][:10]]
        s2_paths = [_rel_to_project(r["path"]) for r in recs["s2"][:10]]
        sample_split[label] = {
            "enroll": s1_paths[:5],
            "dev":    s1_paths[5:10],
            "test":   s2_paths[:10],
        }
        # Tongji palm_id encodes a unique palm; subject identity is not exposed in
        # the dataset, so we record subject == palm_id and side='unknown' for the
        # validator. subject-disjoint is set False because we cannot link L/R.
        label_to_subject_side[label] = {"subject": str(pid), "side": "unknown"}
        if pid in base_set:
            base_ids.append(label)
        elif pid in future_set:
            future_ids.append(label)
        else:
            external_ids.append(label)

    txt_records: List[Tuple[str, int]] = []
    for lbl, parts in sample_split.items():
        for split_name in ("enroll", "dev", "test"):
            for p in parts[split_name]:
                txt_records.append((p, lbl))

    write_all_txt(out_dir, txt_records)
    write_identity_split(
        out_dir,
        base_ids=base_ids, future_ids=future_ids, external_ids=external_ids,
        base_subjects=[str(x) for x in base_palms],
        future_subjects=[str(x) for x in future_palms],
        external_subjects=[str(x) for x in external_palms],
    )
    write_sample_split(out_dir, sample_split)
    write_metadata(
        out_dir,
        dataset_name="tongji_roi",
        identity_unit="palm_side",
        # Tongji image-numbering does not let us link L/R of the same subject,
        # so subject-disjoint is "not enforceable from filenames" — recorded false.
        subject_disjoint=False,
        label_to_subject_side=label_to_subject_side,
        eligible_subject_count=len(eligible_palm_ids),
        eligible_palm_side_count=len(eligible_palm_ids),
        selected_subject_count=len(canonical_palm_order),
        selected_palm_side_count=len(canonical_palm_order),
        base_count=len(base_ids),
        future_count=len(future_ids),
        external_count=len(external_ids),
        sample_split_policy=(
            "Tongji ROI appendix: enroll=session1[:5], dev=session1[5:10], "
            "test=session2[:10] sorted by image_number. RepViT-only because "
            "the CCNet checkpoint was pretrained on Tongji."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_palm_ids), seed={args.seed}); "
            "labels assigned in palm-id order across base sorted, future shuffle, external sorted."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[Tongji] manifest validation passed: {out_dir}")

    # RepViT-only YAML: skip CCNet emission entirely.
    write_configs(
        out_dir,
        dataset_name="tongji_roi",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[100, 200, 300, 400],
        backbones=("repvit",),
        seed=args.seed,
        n_enroll=5, n_dev=5, n_test_min=10,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[Tongji] eligible palm-sides: {len(eligible_palm_ids)}")
    print(f"[Tongji] selected: {len(canonical_palm_order)} — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[Tongji] outputs in: {out_dir} (RepViT-only YAML)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
