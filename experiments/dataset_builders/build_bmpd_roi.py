"""Build the Exp1 manifest for BMPD ROI (appendix: mobile two-session).

Input layout (from ``--root``)::

    BMPD_ROI/
        001/
            001_F_R_0.JPG ... 001_F_R_9.JPG
            001_S_R_10.JPG ... 001_S_R_19.JPG
            001_S_L_20.JPG ... 001_S_L_29.JPG
            001_F_L_30.JPG ... 001_F_L_39.JPG
        002/
            ...

Filename grammar: ``<subject>_<session>_<hand>_<rep>.<ext>`` where session is
F (first session) or S (second session) and hand is L or R. ROI assumed to be
already extracted upstream (this builder does not crop).

Eligibility: palm-side complete iff F session has 10 images AND S session has
10 images. Subject eligible iff BOTH L and R palm-sides complete.

Split: 10 / 20 / 11 subjects = 20 / 40 / 22 palm-sides.
Sample split (per palm-side):
    enroll = F session, all 10
    dev    = S session, first 5 (sorted by rep_idx)
    test   = S session, last 5

Identity: palm-side. Subject-disjoint mandatory.
``future_batch_size = 10``, ``static_gallery_sizes = [20, 30, 40, 50, 60]``.

Note: small dataset → encourage 1% AND 5% FPIR + bootstrap CI in summary.
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


_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<session>[FS])_(?P<hand>[LR])_(?P<rep>\d+)$"
)


def _parse(p: Path):
    if not is_image_file(p):
        return None
    m = _FILENAME_RE.match(p.stem)
    if not m:
        return None
    return {
        "path": p,
        "subject": m.group("subject"),
        "session": m.group("session"),
        "hand": m.group("hand"),
        "rep": int(m.group("rep")),
    }


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build BMPD ROI manifest for Exp1 (appendix)")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_subjects", default=10, type=int)
    parser.add_argument("--n_future_subjects", default=20, type=int)
    parser.add_argument("--n_external_subjects", default=11, type=int)
    parser.add_argument("--future_batch_size", default=10, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_palmside: Dict[Tuple[str, str], Dict[str, List[Dict]]] = defaultdict(
        lambda: {"F": [], "S": []}
    )
    excluded_items: List[Dict] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        rec = _parse(p)
        if rec is None:
            excluded_items.append({"reason": "filename_parse_fail", "path": str(p)})
            continue
        by_palmside[(rec["subject"], rec["hand"])][rec["session"]].append(rec)

    complete_palmsides = set()
    for key, parts in by_palmside.items():
        n_f = len(parts["F"]); n_s = len(parts["S"])
        if n_f >= 10 and n_s >= 10:
            complete_palmsides.add(key)
        else:
            excluded_items.append({
                "reason": "incomplete_palm_side",
                "subject": key[0], "hand": key[1],
                "n_F": n_f, "n_S": n_s,
            })

    subject_to_hands: Dict[str, set] = defaultdict(set)
    for (subject, hand) in complete_palmsides:
        subject_to_hands[subject].add(hand)
    eligible_subjects = sorted(
        s for s, hs in subject_to_hands.items() if {"L", "R"}.issubset(hs)
    )
    for subject, hands in subject_to_hands.items():
        if hands != {"L", "R"}:
            excluded_items.append({
                "reason": "main_split_requires_both_hands_complete",
                "subject": subject, "complete_hands": sorted(hands),
            })

    n_base, n_future, n_external = (
        args.n_base_subjects, args.n_future_subjects, args.n_external_subjects
    )
    total = n_base + n_future + n_external
    if len(eligible_subjects) < total:
        raise RuntimeError(
            f"Need {total} eligible subjects but only {len(eligible_subjects)} have both hands complete."
        )

    rng = np.random.RandomState(args.seed)
    shuffled = list(rng.permutation(eligible_subjects))
    selected = shuffled[:total]
    base_subjects = sorted(selected[:n_base])
    future_subjects = list(selected[n_base:n_base + n_future])
    external_subjects = sorted(selected[n_base + n_future:total])

    canonical_subject_order = list(base_subjects) + list(future_subjects) + list(external_subjects)
    base_set = set(base_subjects)
    future_set = set(future_subjects)

    label_to_subject_side: Dict[int, Dict[str, str]] = {}
    sample_split: Dict[int, Dict[str, List[str]]] = {}
    base_ids: List[int] = []
    future_ids: List[int] = []
    external_ids: List[int] = []
    label = 0

    for subject in canonical_subject_order:
        for hand in ("R", "L"):
            parts = by_palmside[(subject, hand)]
            f_recs = sorted(parts["F"], key=lambda r: r["rep"])[:10]
            s_recs = sorted(parts["S"], key=lambda r: r["rep"])[:10]
            sample_split[label] = {
                "enroll": [_rel_to_project(r["path"]) for r in f_recs],
                "dev":    [_rel_to_project(r["path"]) for r in s_recs[:5]],
                "test":   [_rel_to_project(r["path"]) for r in s_recs[5:10]],
            }
            label_to_subject_side[label] = {"subject": subject, "side": hand}
            if subject in base_set:
                base_ids.append(label)
            elif subject in future_set:
                future_ids.append(label)
            else:
                external_ids.append(label)
            label += 1

    txt_records: List[Tuple[str, int]] = []
    for lbl, parts in sample_split.items():
        for split_name in ("enroll", "dev", "test"):
            for p in parts[split_name]:
                txt_records.append((p, lbl))

    write_all_txt(out_dir, txt_records)
    write_identity_split(
        out_dir,
        base_ids=base_ids, future_ids=future_ids, external_ids=external_ids,
        base_subjects=base_subjects, future_subjects=future_subjects,
        external_subjects=external_subjects,
    )
    write_sample_split(out_dir, sample_split)
    write_metadata(
        out_dir,
        dataset_name="bmpd_roi",
        identity_unit="palm_side",
        subject_disjoint=True,
        label_to_subject_side=label_to_subject_side,
        eligible_subject_count=len(eligible_subjects),
        eligible_palm_side_count=2 * len(eligible_subjects),
        selected_subject_count=len(canonical_subject_order),
        selected_palm_side_count=2 * len(canonical_subject_order),
        base_count=len(base_ids),
        future_count=len(future_ids),
        external_count=len(external_ids),
        sample_split_policy=(
            "BMPD ROI appendix: enroll=F session all 10 (numeric rep_idx sort), "
            "dev=S session [:5], test=S session [5:10]. "
            "Subject-disjoint mandatory; both L and R must be complete."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "labels assigned in (subject_shuffle_order, hand=R then L)."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[BMPD ROI] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name="bmpd_roi",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[20, 30, 40, 50, 60],
        seed=args.seed,
        n_enroll=10, n_dev=5, n_test_min=5,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[BMPD ROI] eligible subjects: {len(eligible_subjects)}")
    print(f"[BMPD ROI] selected: {len(canonical_subject_order)} subjects "
          f"({len(canonical_subject_order)*2} palm-sides) — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[BMPD ROI] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
