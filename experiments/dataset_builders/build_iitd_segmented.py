"""Build the Exp1 manifest for IITD Segmented (main dataset).

Input layout (from ``--root``)::

    IITD/Segmented/
        Right/   (or "Right Hand/")
            001_1.bmp ... 001_5.bmp
            002_1.bmp ...
        Left/    (or "Left Hand/")
            001_1.bmp ...

Filename grammar: ``<subject>_<rep>.<ext>``. Each side is one palm-side
identity per subject; same subject's L and R are distinct palm-sides but
must stay subject-disjoint across base/future/external buckets.

Eligibility: subject must have BOTH Right/<sub>_*.bmp and Left/<sub>_*.bmp
with at least 5 images each.

Split: 25 / 125 / 75 subjects = 50 / 250 / 150 palm-sides. Aborts if eligible
< 225.

Sample split per side (sorted numerically by parsed rep_idx):
    enroll = first 2, dev = next 1, test = remaining (≥2).

Label encoding within the canonical (base sorted, future shuffle, external
sorted) subject order:
    right_label = 2 * subject_idx
    left_label  = 2 * subject_idx + 1
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


_FILENAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<rep>\d+)$")
_RIGHT_DIRS = ("Right", "Right Hand", "right", "right_hand")
_LEFT_DIRS = ("Left", "Left Hand", "left", "left_hand")


def _resolve_side_dir(root: Path, candidates) -> Path:
    for c in candidates:
        d = root / c
        if d.is_dir():
            return d
    raise FileNotFoundError(
        f"Could not locate side directory under {root}. Tried: {list(candidates)}"
    )


def _parse_filename(p: Path):
    if not is_image_file(p):
        return None
    m = _FILENAME_RE.match(p.stem)
    if not m:
        return None
    return {
        "path": p,
        "subject": m.group("subject"),
        "rep": int(m.group("rep")),
    }


def _walk_side(side_dir: Path):
    out = defaultdict(list)
    for p in sorted(side_dir.rglob("*")):
        if not p.is_file():
            continue
        rec = _parse_filename(p)
        if rec is None:
            continue
        out[rec["subject"]].append(rec)
    return out


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build IITD Segmented manifest for Exp1")
    parser.add_argument("--root", required=True, type=str,
                        help="Path to IITD Segmented (containing Right/ and Left/)")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_subjects", default=25, type=int)
    parser.add_argument("--n_future_subjects", default=125, type=int)
    parser.add_argument("--n_external_subjects", default=75, type=int)
    parser.add_argument("--future_batch_size", default=50, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    right_dir = _resolve_side_dir(root, _RIGHT_DIRS)
    left_dir = _resolve_side_dir(root, _LEFT_DIRS)
    right_by_subject = _walk_side(right_dir)
    left_by_subject = _walk_side(left_dir)

    excluded_items: List[Dict] = []

    # Subject is eligible iff both sides have ≥5 images.
    all_subjects = sorted(set(right_by_subject) | set(left_by_subject))
    eligible_subjects: List[str] = []
    for s in all_subjects:
        n_right = len(right_by_subject.get(s, []))
        n_left = len(left_by_subject.get(s, []))
        if n_right >= 5 and n_left >= 5:
            eligible_subjects.append(s)
        else:
            excluded_items.append({
                "reason": "subject_missing_or_short_side",
                "subject": s, "n_right": n_right, "n_left": n_left,
            })

    if len(eligible_subjects) < 225:
        raise RuntimeError(
            f"IITD Segmented main split requires ≥225 eligible subjects but "
            f"only found {len(eligible_subjects)}. Cannot proceed."
        )

    n_base, n_future, n_external = (
        args.n_base_subjects, args.n_future_subjects, args.n_external_subjects
    )
    total = n_base + n_future + n_external
    if len(eligible_subjects) < total:
        raise RuntimeError(
            f"Need {total} subjects but only {len(eligible_subjects)} eligible."
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

    def per_side_split(records):
        recs = sorted(records, key=lambda r: r["rep"])
        paths = [_rel_to_project(r["path"]) for r in recs]
        return {
            "enroll": paths[:2],
            "dev": paths[2:3],
            "test": paths[3:],  # ≥2 by eligibility (n>=5 ⇒ test ≥2)
        }

    # Per the plan: right_label = 2*idx (R first), left_label = 2*idx+1 (L second)
    for subject in canonical_subject_order:
        for side in ("R", "L"):
            recs = right_by_subject[subject] if side == "R" else left_by_subject[subject]
            sample_split[label] = per_side_split(recs)
            label_to_subject_side[label] = {"subject": subject, "side": side}
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
        dataset_name="iitd_segmented",
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
            "IITD Segmented main: per-side sorted by rep_idx; "
            "enroll=[:2], dev=[2:3], test=[3:]. Subject-disjoint enforced."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "labels assigned in (subject_shuffle_order, side=R then L) order."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[IITD] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name="iitd_segmented",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[50, 100, 200, 300],
        seed=args.seed,
        n_enroll=2, n_dev=1, n_test_min=2,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[IITD] eligible subjects: {len(eligible_subjects)} "
          f"(palm-sides: {2 * len(eligible_subjects)})")
    print(f"[IITD] selected: {len(canonical_subject_order)} subjects "
          f"({len(canonical_subject_order)*2} palm-sides) — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[IITD] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
