"""Strict validator for Exp1 dataset manifests.

Every dataset builder must call ``validate_manifest(out_dir)`` after writing
its outputs. The validator raises ManifestValidationError on the first failure
class with a descriptive message, but collects all failures within a class
before raising so the user can fix several issues per round-trip.

Manifest paths are stored relative to the project root. ``project_root`` is
inferred from ``out_dir`` when it lives at ``<project_root>/experiments/
generated/<dataset_name>``; otherwise the caller must pass ``project_root``
explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ManifestValidationError(Exception):
    """Raised when a manifest fails validation. Message lists the failures."""


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_project_root(out_dir: Path) -> Path:
    """Default: out_dir = <root>/experiments/generated/<name> → root.

    Falls back to out_dir.resolve() if the layout doesn't match.
    """
    out_dir = out_dir.resolve()
    parts = out_dir.parts
    if len(parts) >= 3 and parts[-3] == "experiments" and parts[-2] == "generated":
        return Path(*parts[:-3])
    return out_dir


def _check(condition: bool, errors: List[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_manifest(
    out_dir,
    project_root: Optional[Path] = None,
) -> None:
    """Validate the four manifest files in ``out_dir``.

    Raises ManifestValidationError on any failure. Returns silently on success.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise ManifestValidationError(f"out_dir does not exist: {out_dir}")
    project_root = Path(project_root) if project_root else _infer_project_root(out_dir)

    all_txt = out_dir / "all.txt"
    id_split_path = out_dir / "identity_split.json"
    sample_split_path = out_dir / "sample_split.json"
    metadata_path = out_dir / "metadata.json"

    missing = [p.name for p in (all_txt, id_split_path, sample_split_path, metadata_path) if not p.exists()]
    if missing:
        raise ManifestValidationError(f"Missing manifest files in {out_dir}: {missing}")

    id_split = _load_json(id_split_path)
    sample_split = _load_json(sample_split_path)
    metadata = _load_json(metadata_path)

    errors: List[str] = []

    base_ids = list(id_split.get("base_ids", []))
    future_ids = list(id_split.get("future_ids", []))
    external_ids = list(id_split.get("external_ids", []))
    base_set = set(base_ids)
    future_set = set(future_ids)
    external_set = set(external_ids)

    # 1. base/future/external id overlap
    overlap_bf = base_set & future_set
    overlap_be = base_set & external_set
    overlap_fe = future_set & external_set
    if overlap_bf:
        errors.append(f"base/future overlap: {sorted(overlap_bf)[:10]}")
    if overlap_be:
        errors.append(f"base/external overlap: {sorted(overlap_be)[:10]}")
    if overlap_fe:
        errors.append(f"future/external overlap: {sorted(overlap_fe)[:10]}")

    # 2. labels in sample_split must lie in identity_split
    selected = base_set | future_set | external_set
    sample_labels: List[int] = []
    for k in sample_split.keys():
        try:
            sample_labels.append(int(k))
        except ValueError:
            errors.append(f"sample_split key not parseable as int: {k!r}")
    sample_label_set = set(sample_labels)
    unknown_labels = sample_label_set - selected
    if unknown_labels:
        errors.append(
            f"sample_split contains labels not in identity_split: "
            f"{sorted(unknown_labels)[:10]}"
        )
    missing_labels = selected - sample_label_set
    if missing_labels:
        errors.append(
            f"identity_split labels missing from sample_split: "
            f"{sorted(missing_labels)[:10]}"
        )

    # 3 + 4 + 5. per-identity enroll/dev/test sanity
    for label in sorted(selected & sample_label_set):
        parts = sample_split.get(str(label), {})
        enroll = parts.get("enroll", [])
        dev = parts.get("dev", [])
        test = parts.get("test", [])
        if not enroll:
            errors.append(f"label {label}: empty enroll list")
        if not dev:
            errors.append(f"label {label}: empty dev list")
        if not test:
            errors.append(f"label {label}: empty test list")
        if len(test) < 2:
            errors.append(f"label {label}: len(test)={len(test)} < 2")
        e_set, d_set, t_set = set(enroll), set(dev), set(test)
        if e_set & d_set:
            errors.append(f"label {label}: enroll/dev overlap: {sorted(e_set & d_set)[:5]}")
        if e_set & t_set:
            errors.append(f"label {label}: enroll/test overlap: {sorted(e_set & t_set)[:5]}")
        if d_set & t_set:
            errors.append(f"label {label}: dev/test overlap: {sorted(d_set & t_set)[:5]}")

    # 6. image paths must exist on disk
    missing_paths: List[str] = []
    for label_str, parts in sample_split.items():
        for split_name in ("enroll", "dev", "test"):
            for rel_path in parts.get(split_name, []):
                abs_path = (project_root / rel_path).resolve()
                if not abs_path.is_file():
                    missing_paths.append(rel_path)
                    if len(missing_paths) >= 20:
                        break
            if len(missing_paths) >= 20:
                break
        if len(missing_paths) >= 20:
            break
    if missing_paths:
        errors.append(
            f"{len(missing_paths)} image path(s) not found on disk under "
            f"project_root={project_root}; first: {missing_paths[:5]}"
        )

    # 7. subject_disjoint: same subject must not span buckets
    subject_disjoint = bool(metadata.get("subject_disjoint", False))
    if subject_disjoint:
        label_to_subject_side = metadata.get("label_to_subject_side", {})
        subject_to_buckets: Dict[str, set] = {}
        for label_str, info in label_to_subject_side.items():
            try:
                label = int(label_str)
            except ValueError:
                continue
            subject = info.get("subject")
            if subject is None:
                continue
            if label in base_set:
                bucket = "base"
            elif label in future_set:
                bucket = "future"
            elif label in external_set:
                bucket = "external"
            else:
                continue
            subject_to_buckets.setdefault(subject, set()).add(bucket)
        bad_subjects = {s: sorted(b) for s, b in subject_to_buckets.items() if len(b) > 1}
        if bad_subjects:
            sample = dict(list(bad_subjects.items())[:5])
            errors.append(
                f"subject_disjoint=true but {len(bad_subjects)} subject(s) span "
                f"multiple buckets; first: {sample}"
            )

    # 8. labels 0-indexed contiguous over the full selected set
    if selected:
        expected = set(range(len(selected)))
        if selected != expected:
            extra = sorted(selected - expected)[:5]
            missing = sorted(expected - selected)[:5]
            errors.append(
                f"labels not 0-indexed contiguous: extra={extra}, missing={missing}, "
                f"selected_count={len(selected)}"
            )

    # 9. all.txt: only selected identities, paths match sample_split
    txt_records: List[Tuple[str, int]] = []
    with open(all_txt, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                path_part, label_part = line.rsplit(" ", 1)
                txt_records.append((path_part, int(label_part)))
            except ValueError:
                errors.append(f"all.txt line {lineno} unparseable: {line!r}")
    txt_label_set = {lbl for _, lbl in txt_records}
    out_of_split = txt_label_set - selected
    if out_of_split:
        errors.append(
            f"all.txt contains labels not in identity_split: {sorted(out_of_split)[:10]}"
        )
    txt_pairs = {(label, path_part) for path_part, label in txt_records}
    sample_pairs = set()
    for label_str, parts in sample_split.items():
        try:
            label = int(label_str)
        except ValueError:
            continue
        for split_name in ("enroll", "dev", "test"):
            sample_pairs.update((label, path) for path in parts.get(split_name, []))

    extra_in_txt = sorted(txt_pairs - sample_pairs)[:10]
    missing_in_txt = sorted(sample_pairs - txt_pairs)[:10]
    if extra_in_txt:
        errors.append(
            f"extra_in_txt: all.txt has (label, path) pairs not present in sample_split; "
            f"first: {extra_in_txt[:5]}"
        )
    if missing_in_txt:
        errors.append(
            f"missing_in_txt: sample_split has (label, path) pairs not present in all.txt; "
            f"first: {missing_in_txt[:5]}"
        )

    # 10. future_ids must equal metadata.future_subject_order
    future_subject_order = metadata.get("future_subject_order")
    if future_subject_order is None:
        errors.append("metadata.future_subject_order missing")
    else:
        canonical = [int(x) for x in future_subject_order]
        if list(future_ids) != canonical:
            errors.append(
                "identity_split.future_ids != metadata.future_subject_order; "
                f"first divergence at index "
                f"{next((i for i, (a, b) in enumerate(zip(future_ids, canonical)) if a != b), 'len-mismatch')}"
            )

    if errors:
        bullet = "\n  - ".join(errors)
        raise ManifestValidationError(
            f"Manifest validation failed for {out_dir} ({len(errors)} issue(s)):\n  - {bullet}"
        )
