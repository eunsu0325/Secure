"""Build the Exp1 manifest for GPDS150hand (main dataset).

Input layout (from ``--root``)::

    HandsGPDS150/manosGPDS/
        001/
            mano1_1.jpg ... mano1_10.jpg   <-- skipped (whole-hand images)
            palma1_1.jpg ... palma1_10.jpg <-- USED (palm ROI)
        002/
            ...
        150/

The builder uses **only ``palma*.jpg``** files. ``mano*`` (full-hand) files
are excluded explicitly per the experiment design so the comparison with
BJTU/IITD stays honest (palm ROI only).

Filename grammar: ``palma<rep_a>_<rep_b>.<ext>``. The numeric components
together act as a repetition index; we sort by ``(rep_a, rep_b)`` as the
canonical order.

Identity: one right-hand palm per subject (subject-disjoint trivially true
since each subject contributes exactly one palm-side identity).

Split: 30 / 70 / 50 subjects = 30 / 70 / 50 palm-sides.
Sample split: enroll = first 2, dev = next 2, test = remaining ≥6.
``future_batch_size = 10``, ``static_gallery_sizes = [30, 50, 70, 100]``.
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


_PALMA_RE = re.compile(r"^palma(?P<rep_a>\d+)_(?P<rep_b>\d+)$", re.IGNORECASE)


def _resolve_subject_root(root: Path) -> Path:
    """Return the directory containing the per-subject folders (manosGPDS/)."""
    if (root / "manosGPDS").is_dir():
        return root / "manosGPDS"
    # Already pointed at manosGPDS/ ?
    if any(p.is_dir() and p.name.isdigit() for p in root.iterdir()):
        return root
    raise FileNotFoundError(
        f"Cannot locate per-subject folders under {root}. "
        "Expected ``manosGPDS/<NNN>/palma*.jpg`` layout."
    )


def _parse_palma(p: Path):
    if not is_image_file(p):
        return None
    if not p.name.lower().startswith("palma"):
        return None  # Skip mano*.jpg etc.
    m = _PALMA_RE.match(p.stem)
    if not m:
        return None
    return {"path": p, "rep_a": int(m.group("rep_a")), "rep_b": int(m.group("rep_b"))}


def _walk_subject(subject_dir: Path):
    out: List[Dict] = []
    excluded: List[Dict] = []
    for p in sorted(subject_dir.rglob("*")):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        rec = _parse_palma(p)
        if rec is None:
            if p.name.lower().startswith("mano"):
                excluded.append({"reason": "mano_excluded_by_design", "path": str(p)})
            else:
                excluded.append({"reason": "filename_parse_fail", "path": str(p)})
            continue
        out.append(rec)
    return out, excluded


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build GPDS150hand manifest for Exp1")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_subjects", default=30, type=int)
    parser.add_argument("--n_future_subjects", default=70, type=int)
    parser.add_argument("--n_external_subjects", default=50, type=int)
    parser.add_argument("--future_batch_size", default=10, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects_root = _resolve_subject_root(root)

    # Each subject directory → list of palma records
    by_subject: Dict[str, List[Dict]] = {}
    excluded_items: List[Dict] = []
    for subject_dir in sorted(subjects_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        if not subject_dir.name.lstrip("0").isdigit() and not subject_dir.name.isdigit():
            continue
        recs, ex = _walk_subject(subject_dir)
        excluded_items.extend(ex)
        if recs:
            by_subject[subject_dir.name] = recs

    # Eligible subject: ≥ enroll(2) + dev(2) + min_test(6) = 10 palma images.
    eligible_subjects: List[str] = []
    for subject in sorted(by_subject):
        n = len(by_subject[subject])
        if n >= 10:
            eligible_subjects.append(subject)
        else:
            excluded_items.append({
                "reason": "subject_below_min_palma",
                "subject": subject, "n_palma": n,
            })

    n_base, n_future, n_external = (
        args.n_base_subjects, args.n_future_subjects, args.n_external_subjects
    )
    total = n_base + n_future + n_external
    if len(eligible_subjects) < total:
        raise RuntimeError(
            f"Need {total} eligible subjects but only {len(eligible_subjects)} have ≥10 palma images."
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

    for label, subject in enumerate(canonical_subject_order):
        recs = sorted(by_subject[subject], key=lambda r: (r["rep_a"], r["rep_b"]))
        paths = [_rel_to_project(r["path"]) for r in recs]
        sample_split[label] = {
            "enroll": paths[:2],
            "dev": paths[2:4],
            "test": paths[4:],
        }
        # GPDS150hand uses right-hand palms only
        label_to_subject_side[label] = {"subject": subject, "side": "R"}
        if subject in base_set:
            base_ids.append(label)
        elif subject in future_set:
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
        dataset_name="gpds150hand",
        identity_unit="palm_side",
        subject_disjoint=True,
        label_to_subject_side=label_to_subject_side,
        eligible_subject_count=len(eligible_subjects),
        eligible_palm_side_count=len(eligible_subjects),
        selected_subject_count=len(canonical_subject_order),
        selected_palm_side_count=len(canonical_subject_order),
        base_count=len(base_ids),
        future_count=len(future_ids),
        external_count=len(external_ids),
        sample_split_policy=(
            "GPDS150hand main: palma*.jpg only (mano* excluded by design); "
            "sorted numerically by (rep_a, rep_b); "
            "enroll=[:2], dev=[2:4], test=[4:]."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "single right-hand palm-side per subject; one label per subject in canonical order."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[GPDS150hand] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name="gpds150hand",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[30, 50, 70, 100],
        seed=args.seed,
        n_enroll=2, n_dev=2, n_test_min=6,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[GPDS150hand] eligible subjects: {len(eligible_subjects)}")
    print(f"[GPDS150hand] selected: {len(canonical_subject_order)} — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[GPDS150hand] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
