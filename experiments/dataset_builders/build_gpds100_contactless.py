"""Build the Exp1 manifest for GPDS100 Contactless 2-Bands (appendix).

Input layout (from ``--root``)::

    HandsGPDS100Contactless2bands/
        001/
            palma_001_01.jpg ... palma_001_NN.jpg
            visible_001_01.bmp ...    <-- excluded by main Exp1
            Infraro_001_01.bmp ...    <-- excluded by main Exp1
        002/
        ...

Filename grammar for the used files: ``palma_<subject>_<rep>.<ext>``.
``visible_*.bmp`` and ``Infraro_*.bmp`` are excluded — cross-band stress
tests live in a separate pipeline.

Identity: right-hand palm per subject (subject-disjoint trivially true,
one palm-side per subject in main Exp1 mode).

Split: 20 / 40 / 40 subjects.
Sample split: enroll = first 2, dev = next 3, test = remaining ≥5.
``future_batch_size = 10``, ``static_gallery_sizes = [20, 30, 40, 60]``.
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


_PALMA_RE = re.compile(r"^palma_(?P<subject>\d+)_(?P<rep>\d+)$", re.IGNORECASE)


def _parse(p: Path):
    if not is_image_file(p):
        return None
    name = p.stem
    m = _PALMA_RE.match(name)
    if m is None:
        return None
    return {"path": p, "subject": m.group("subject"), "rep": int(m.group("rep"))}


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build GPDS100 Contactless 2-Bands manifest for Exp1 (appendix)"
    )
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base_subjects", default=20, type=int)
    parser.add_argument("--n_future_subjects", default=40, type=int)
    parser.add_argument("--n_external_subjects", default=40, type=int)
    parser.add_argument("--future_batch_size", default=10, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    excluded_items: List[Dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        nm = p.name.lower()
        if nm.startswith("visible") or nm.startswith("infraro"):
            excluded_items.append({"reason": "non_palma_band_excluded", "path": str(p)})
            continue
        rec = _parse(p)
        if rec is None:
            excluded_items.append({"reason": "filename_parse_fail", "path": str(p)})
            continue
        by_subject[rec["subject"]].append(rec)

    eligible_subjects: List[str] = []
    for s in sorted(by_subject):
        n = len(by_subject[s])
        if n >= 10:  # 2 enroll + 3 dev + ≥5 test = 10
            eligible_subjects.append(s)
        else:
            excluded_items.append({
                "reason": "subject_below_min_palma",
                "subject": s, "n_palma": n,
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
        recs = sorted(by_subject[subject], key=lambda r: r["rep"])
        paths = [_rel_to_project(r["path"]) for r in recs]
        sample_split[label] = {
            "enroll": paths[:2],
            "dev":    paths[2:5],
            "test":   paths[5:],
        }
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
        base_ids=base_ids, future_ids=future_ids, external_ids=external_ids,
        base_subjects=base_subjects, future_subjects=future_subjects,
        external_subjects=external_subjects,
    )
    write_sample_split(out_dir, sample_split)
    write_metadata(
        out_dir,
        dataset_name="gpds100_contactless",
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
            "GPDS100 contactless appendix: palma_*.jpg only "
            "(visible_*.bmp / Infraro_*.bmp excluded by design); "
            "sorted by numeric rep; enroll=[:2], dev=[2:5], test=[5:]."
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "single right-hand palm-side per subject."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[GPDS100C] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name="gpds100_contactless",
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[20, 30, 40, 60],
        seed=args.seed,
        n_enroll=2, n_dev=3, n_test_min=5,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[GPDS100C] eligible subjects: {len(eligible_subjects)}")
    print(f"[GPDS100C] selected: {len(canonical_subject_order)} — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[GPDS100C] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
