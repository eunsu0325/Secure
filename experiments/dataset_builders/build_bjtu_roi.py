"""Build the Exp1 manifest for BJTU PalmV2 ROI (main dataset, 1순위).

Input layout (from ``--root``)::

    BJTU_PalmV2(ROI)/
        train/
            001_F_L1C1.jpg ... 001_F_L5C1.jpg
            001_F_R1C2.jpg ... 001_F_R5C2.jpg
            ...
        test/
            001_S_L1C1.jpg ...
            001_S_R1C2.jpg ...

Filename grammar: ``<subject>_<session>_<hand><rep>C<class>.<ext>`` where
``session ∈ {F, S}``, ``hand ∈ {L, R}``, ``rep`` and ``class`` are integers.
The ``C`` index identifies the palm-side class globally; ``L``/``R`` is the
hand. Together ``(subject, hand)`` defines a palm-side identity.

Eligibility (main split): a subject is eligible iff **both** its left and
right palm-sides have at least 5 train + 5 test images. Subjects with only
one complete palm-side are excluded from the main split (recorded in
``metadata.excluded_items``). This guarantees the 110-subject / 220-palm-side
count is exact and that subject-disjoint holds without ad-hoc tie-breaking.

Sample split per palm-side (sorted by parsed numeric rep_idx):
    * enroll = train[:3]
    * dev    = train[3:5]
    * test   = test[:5]

Identity split: 15 / 60 / 35 subjects = 30 / 120 / 70 palm-sides.
``future_batch_size = 10``, ``static_gallery_sizes = [30, 60, 100, 150]``.
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
    r"^(?P<subject>\d+)_(?P<session>[FS])_(?P<hand>[LR])(?P<rep>\d+)C(?P<klass>\d+)$"
)


def _parse_filename(p: Path):
    """Return parsed fields or None if the filename doesn't match the grammar."""
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
        "c_class": int(m.group("klass")),
    }


def _walk(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return [p for p in sorted(root.rglob("*")) if p.is_file() and is_image_file(p)]


def _rel_to_project(path: Path) -> str:
    """Return path as a string relative to PROJECT_ROOT if possible, else absolute."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build BJTU PalmV2 ROI manifest for Exp1")
    parser.add_argument("--root", required=True, type=str,
                        help="Path to BJTU_PalmV2(ROI) containing train/ and test/")
    parser.add_argument("--version", default="v2", choices=["v2"],
                        help="Currently only V2 is supported")
    parser.add_argument("--out", required=True, type=str,
                        help="Output directory (e.g. experiments/generated/bjtu_v2)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_subjects", default=15, type=int)
    parser.add_argument("--n_future_subjects", default=60, type=int)
    parser.add_argument("--n_external_subjects", default=35, type=int)
    parser.add_argument("--future_batch_size", default=10, type=int)
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip validate_manifest at the end (for diagnostic builds)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = _walk(root / "train")
    test_files = _walk(root / "test")
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"BJTU root must contain train/ and test/ with images: {root}"
        )

    # Group records by (subject, hand) and by kind (train/test)
    by_palmside: Dict[Tuple[str, str], Dict[str, List[Dict]]] = defaultdict(
        lambda: {"train": [], "test": []}
    )
    excluded_items: List[Dict] = []
    for kind, files in (("train", train_files), ("test", test_files)):
        for p in files:
            rec = _parse_filename(p)
            if rec is None:
                excluded_items.append({"reason": "filename_parse_fail", "path": str(p)})
                continue
            by_palmside[(rec["subject"], rec["hand"])][kind].append(rec)

    # Eligible palm-side: ≥5 train AND ≥5 test images.
    complete_palmsides = set()
    for key, parts in by_palmside.items():
        if len(parts["train"]) >= 5 and len(parts["test"]) >= 5:
            complete_palmsides.add(key)
        else:
            excluded_items.append({
                "reason": "incomplete_palm_side",
                "subject": key[0], "hand": key[1],
                "n_train": len(parts["train"]), "n_test": len(parts["test"]),
            })

    # Eligible subject (main split): BOTH L and R palm-sides complete.
    subject_to_hands: Dict[str, set] = defaultdict(set)
    for (subject, hand) in complete_palmsides:
        subject_to_hands[subject].add(hand)
    eligible_subjects = sorted(
        s for s, hs in subject_to_hands.items() if {"L", "R"}.issubset(hs)
    )
    eligible_palm_side_count = 2 * len(eligible_subjects)

    # Subjects with only one complete palm-side are excluded from main split.
    for subject, hands in subject_to_hands.items():
        if hands != {"L", "R"}:
            excluded_items.append({
                "reason": "main_split_requires_both_hands_complete",
                "subject": subject,
                "complete_hands": sorted(hands),
            })

    n_base = args.n_base_subjects
    n_future = args.n_future_subjects
    n_external = args.n_external_subjects
    total = n_base + n_future + n_external
    if len(eligible_subjects) < total:
        raise RuntimeError(
            f"Need {total} eligible subjects but only {len(eligible_subjects)} have "
            "both L and R complete palm-sides."
        )

    rng = np.random.RandomState(args.seed)
    shuffled = list(rng.permutation(eligible_subjects))
    selected = shuffled[:total]
    base_subjects = sorted(selected[:n_base])
    future_subjects = list(selected[n_base:n_base + n_future])  # SHUFFLE ORDER
    external_subjects = sorted(selected[n_base + n_future:total])
    selected_subject_count = len(base_subjects) + len(future_subjects) + len(external_subjects)
    selected_palm_side_count = 2 * selected_subject_count

    # Assign labels in canonical order (base sorted, future shuffle, external sorted),
    # within each subject (L, R). 0-indexed contiguous globally.
    canonical_subject_order = list(base_subjects) + list(future_subjects) + list(external_subjects)
    label_to_subject_side: Dict[int, Dict[str, str]] = {}
    sample_split: Dict[int, Dict[str, List[str]]] = {}
    label = 0
    base_ids: List[int] = []
    future_ids: List[int] = []
    external_ids: List[int] = []
    base_set = set(base_subjects)
    future_set = set(future_subjects)
    for subject in canonical_subject_order:
        for hand in ("L", "R"):
            recs = by_palmside[(subject, hand)]
            train_recs = sorted(recs["train"], key=lambda r: r["rep"])
            test_recs = sorted(recs["test"], key=lambda r: r["rep"])
            enroll = [_rel_to_project(r["path"]) for r in train_recs[:3]]
            dev = [_rel_to_project(r["path"]) for r in train_recs[3:5]]
            test = [_rel_to_project(r["path"]) for r in test_recs[:5]]
            sample_split[label] = {"enroll": enroll, "dev": dev, "test": test}
            label_to_subject_side[label] = {"subject": subject, "side": hand}
            if subject in base_set:
                base_ids.append(label)
            elif subject in future_set:
                future_ids.append(label)
            else:
                external_ids.append(label)
            label += 1

    # Records for all.txt
    txt_records: List[Tuple[str, int]] = []
    for lbl, parts in sample_split.items():
        for split_name in ("enroll", "dev", "test"):
            for p in parts[split_name]:
                txt_records.append((p, lbl))

    write_all_txt(out_dir, txt_records)
    write_identity_split(
        out_dir,
        base_ids=base_ids,
        future_ids=future_ids,
        external_ids=external_ids,
        base_subjects=base_subjects,
        future_subjects=future_subjects,
        external_subjects=external_subjects,
    )
    write_sample_split(out_dir, sample_split)
    write_metadata(
        out_dir,
        dataset_name="bjtu_v2",
        identity_unit="palm_side",
        subject_disjoint=True,
        label_to_subject_side=label_to_subject_side,
        eligible_subject_count=len(eligible_subjects),
        eligible_palm_side_count=eligible_palm_side_count,
        selected_subject_count=selected_subject_count,
        selected_palm_side_count=selected_palm_side_count,
        base_count=len(base_ids),
        future_count=len(future_ids),
        external_count=len(external_ids),
        sample_split_policy=(
            "BJTU V2 main: enroll=train[:3], dev=train[3:5], test=test[:5] "
            "sorted numerically by parsed rep_idx (L1<L2<...<L10). "
            "Subject-disjoint enforced; main split requires both L and R complete."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "labels assigned in (subject_shuffle_order, hand=L,R) order, "
            "so future_ids ascends contiguously while subject mapping reflects the shuffle."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[BJTU V2] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name="bjtu_v2",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[30, 60, 100, 150],
        seed=args.seed,
        n_enroll=3, n_dev=2, n_test_min=5,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[BJTU V2] eligible subjects: {len(eligible_subjects)} "
          f"(palm-sides: {eligible_palm_side_count})")
    print(f"[BJTU V2] selected: {selected_subject_count} subjects "
          f"({selected_palm_side_count} palm-sides) — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[BJTU V2] excluded items: {len(excluded_items)}")
    print(f"[BJTU V2] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
