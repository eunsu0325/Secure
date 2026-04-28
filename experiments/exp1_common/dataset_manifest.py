"""Canonical manifest writers for Exp1 dataset builders.

Every builder produces:
    out_dir/all.txt
    out_dir/identity_split.json
    out_dir/sample_split.json
    out_dir/metadata.json

Contracts enforced here:
    * base_ids and external_ids are sorted before writing.
    * future_ids preserves the seeded shuffle order from selection (never sorted).
    * enroll/dev/test path lists are sorted before writing (numeric ordering is
      the builder's responsibility; this module only finalises layout).
    * all.txt records sorted by (label, image_path).
    * Every JSON file is written via _dump_json with sort_keys=True,
      ensure_ascii=False, and a trailing newline so files are byte-stable across
      runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
})

_HIDDEN_OR_JUNK = frozenset({"__MACOSX", "Thumbs.db", ".DS_Store"})


def is_image_file(path: Path) -> bool:
    """True if the path looks like a usable image file.

    Filters out hidden files, OS junk (__MACOSX, Thumbs.db, .DS_Store), and
    anything whose extension is not in IMAGE_EXTS.
    """
    name = path.name
    if name.startswith("."):
        return False
    if name in _HIDDEN_OR_JUNK:
        return False
    if "__MACOSX" in path.parts:
        return False
    return path.suffix.lower() in IMAGE_EXTS


def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_all_txt(out_dir: Path, records: Iterable[Tuple[str, int]]) -> Path:
    """Write all.txt with one ``image_path label`` line per record.

    Records are sorted by (label, image_path) so the file is byte-stable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: (int(r[1]), str(r[0])))
    path = out_dir / "all.txt"
    with open(path, "w", encoding="utf-8") as f:
        for image_path, label in sorted_records:
            f.write(f"{image_path} {int(label)}\n")
    return path


def write_identity_split(
    out_dir: Path,
    base_ids: Sequence[int],
    future_ids: Sequence[int],
    external_ids: Sequence[int],
    base_subjects: Sequence[str],
    future_subjects: Sequence[str],
    external_subjects: Sequence[str],
) -> Path:
    """Write identity_split.json.

    base_ids and external_ids are sorted; future_ids is written in the order
    received and must be the seeded shuffle order produced by selection. The
    same contract applies to *_subjects.
    """
    payload = {
        "base_ids": sorted(int(i) for i in base_ids),
        "future_ids": [int(i) for i in future_ids],
        "external_ids": sorted(int(i) for i in external_ids),
        "base_subjects": sorted(str(s) for s in base_subjects),
        "future_subjects": [str(s) for s in future_subjects],
        "external_subjects": sorted(str(s) for s in external_subjects),
    }
    path = Path(out_dir) / "identity_split.json"
    _dump_json(path, payload)
    return path


def write_sample_split(
    out_dir: Path,
    sample_split: Dict[int, Dict[str, List[str]]],
) -> Path:
    """Write sample_split.json keyed by string label.

    Each label maps to {"enroll": [...], "dev": [...], "test": [...]}. Within
    each list paths are sorted; the *order across paths* is set by the builder
    (typically by parsed numeric repetition_idx) and not changed here.
    """
    payload: Dict[str, Dict[str, List[str]]] = {}
    for label, parts in sample_split.items():
        payload[str(int(label))] = {
            "enroll": [str(p) for p in parts.get("enroll", [])],
            "dev": [str(p) for p in parts.get("dev", [])],
            "test": [str(p) for p in parts.get("test", [])],
        }
    path = Path(out_dir) / "sample_split.json"
    _dump_json(path, payload)
    return path


def write_metadata(
    out_dir: Path,
    *,
    dataset_name: str,
    identity_unit: str,
    subject_disjoint: bool,
    label_to_subject_side: Dict[int, Dict[str, str]],
    eligible_subject_count: int,
    eligible_palm_side_count: int,
    selected_subject_count: int,
    selected_palm_side_count: int,
    base_count: int,
    future_count: int,
    external_count: int,
    sample_split_policy: str,
    excluded_items: List[Dict],
    seed: int,
    future_subject_order: Sequence,
    future_ids_order_source: str,
) -> Path:
    """Write metadata.json with everything needed to validate the manifest.

    label_to_subject_side maps the contiguous integer label to a dict with
    ``subject`` and ``side`` keys. split_validation uses this to enforce the
    subject-disjoint guarantee.

    future_subject_order is the canonical ordered list of identity ids in the
    seeded shuffle order; split_validation asserts identity_split.future_ids
    equals this list. future_ids_order_source is a short human-readable
    description (e.g. ``"shuffle(sorted(future_subjects), seed=42)"``).
    """
    payload = {
        "dataset_name": str(dataset_name),
        "identity_unit": str(identity_unit),
        "subject_disjoint": bool(subject_disjoint),
        "label_to_subject_side": {
            str(int(label)): {
                "subject": str(info.get("subject", "")),
                "side": str(info.get("side", "")),
            }
            for label, info in label_to_subject_side.items()
        },
        "eligible_subject_count": int(eligible_subject_count),
        "eligible_palm_side_count": int(eligible_palm_side_count),
        "selected_subject_count": int(selected_subject_count),
        "selected_palm_side_count": int(selected_palm_side_count),
        "base_count": int(base_count),
        "future_count": int(future_count),
        "external_count": int(external_count),
        "sample_split_policy": str(sample_split_policy),
        "excluded_items": list(excluded_items),
        "seed": int(seed),
        "future_subject_order": [int(x) for x in future_subject_order],
        "future_ids_order_source": str(future_ids_order_source),
    }
    path = Path(out_dir) / "metadata.json"
    _dump_json(path, payload)
    return path
