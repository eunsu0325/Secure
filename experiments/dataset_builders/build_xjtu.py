"""Build the Exp1 manifest for XJTU (appendix: cross-illumination / cross-device).

Input layout::

    huawei/
        Flash/
            L_001/  *.jpg ...
            R_001/  *.jpg ...
        Nature/
            L_001/  *.jpg ...
            R_001/  *.jpg ...

    iPhone/
        Flash/
            L_001/  ...
        Nature/
            L_001/  ...

Identity: palm-side; ``L_001`` and ``R_001`` are distinct palm-sides of
subject ``001``. Subject-disjoint enforced.

Two CLI modes:

  * ``--mode flash_to_nature`` (default; primary): only Huawei is required.
    Eligibility: Huawei has Flash 10 + Nature 10 for that palm-side.
    Sample split per palm-side: enroll = Flash[:5], dev = Flash[5:10],
    test = Nature[:10].

  * ``--mode cross_device``: both Huawei AND iPhone must be complete (10
    Flash + 10 Nature each). Sample split per palm-side: enroll =
    Huawei-Flash[:5], dev = Huawei-Flash[5:10], test = iPhone-Nature[:10].

Recommended split: 15 / 50 / 25 subjects = 30 / 100 / 50 palm-sides.
``future_batch_size = 10``, ``static_gallery_sizes = [30, 60, 100, 130]``.
The static-gallery sizes are quoted in palm-side counts to match the BJTU/IITD
convention; the minimum equals base palm-sides (15 subjects × 2 = 30).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


_DIR_RE = re.compile(r"^(?P<hand>[LR])_(?P<subject>\d+)$")


def _walk_palmside_dirs(device_root: Path):
    """device_root/<illum>/<L_or_R>_<subject>/*.<ext>

    Returns nested dict: {(subject, hand): {illum: [paths sorted by name]}}.
    """
    out: Dict[Tuple[str, str], Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    if not device_root.is_dir():
        return out
    for illum_dir in sorted(device_root.iterdir()):
        if not illum_dir.is_dir():
            continue
        illum = illum_dir.name
        for palmside_dir in sorted(illum_dir.iterdir()):
            if not palmside_dir.is_dir():
                continue
            m = _DIR_RE.match(palmside_dir.name)
            if not m:
                continue
            subject = m.group("subject")
            hand = m.group("hand")
            files = [p for p in sorted(palmside_dir.rglob("*"))
                     if p.is_file() and is_image_file(p)]
            # Numeric sort fallback by trailing-int in stem; otherwise lex.
            def sort_key(p: Path):
                m_n = re.search(r"(\d+)\s*$", p.stem)
                return (0, int(m_n.group(1))) if m_n else (1, p.name)
            files.sort(key=sort_key)
            out[(subject, hand)][illum] = files
    return out


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build XJTU manifest for Exp1 (appendix)")
    parser.add_argument("--huawei_root", required=True, type=str,
                        help="Path to .../huawei/ containing Flash/ and Nature/")
    parser.add_argument("--iphone_root", default=None, type=str,
                        help="Path to .../iPhone/ — required for --mode cross_device")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mode", default="flash_to_nature",
                        choices=["flash_to_nature", "cross_device"])
    parser.add_argument("--n_base_subjects", default=15, type=int,
                        help="Base subjects; palm-sides = 2× subjects (default 15→30 palm-sides)")
    parser.add_argument("--n_future_subjects", default=50, type=int,
                        help="Future subjects; palm-sides = 2× subjects (default 50→100)")
    parser.add_argument("--n_external_subjects", default=25, type=int,
                        help="External subjects; palm-sides = 2× subjects (default 25→50)")
    parser.add_argument("--future_batch_size", default=10, type=int)
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args(argv)

    huawei_root = Path(args.huawei_root)
    iphone_root = Path(args.iphone_root) if args.iphone_root else None
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    if out_dir.name == "xjtu":
        raise ValueError(
            "--out experiments/generated/xjtu is reserved; use "
            "experiments/generated/xjtu_flash_to_nature or "
            "experiments/generated/xjtu_cross_device"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = f"xjtu_{args.mode}"

    if args.mode == "cross_device" and iphone_root is None:
        raise ValueError("--mode cross_device requires --iphone_root")

    huawei_data = _walk_palmside_dirs(huawei_root)
    iphone_data = _walk_palmside_dirs(iphone_root) if iphone_root else {}

    # Eligible palm-side: depends on mode.
    excluded_items: List[Dict] = []
    eligible_palmsides: List[Tuple[str, str]] = []
    for key in sorted(set(huawei_data) | set(iphone_data)):
        h_flash = huawei_data.get(key, {}).get("Flash", [])
        h_nat = huawei_data.get(key, {}).get("Nature", [])
        i_flash = iphone_data.get(key, {}).get("Flash", []) if iphone_data else []
        i_nat = iphone_data.get(key, {}).get("Nature", []) if iphone_data else []
        if args.mode == "flash_to_nature":
            ok = len(h_flash) >= 10 and len(h_nat) >= 10
        else:  # cross_device
            ok = (len(h_flash) >= 10 and len(h_nat) >= 10
                  and len(i_flash) >= 10 and len(i_nat) >= 10)
        if ok:
            eligible_palmsides.append(key)
        else:
            excluded_items.append({
                "reason": "incomplete_palm_side",
                "subject": key[0], "hand": key[1],
                "n_huawei_flash": len(h_flash), "n_huawei_nature": len(h_nat),
                "n_iphone_flash": len(i_flash), "n_iphone_nature": len(i_nat),
                "mode": args.mode,
            })

    # Eligible subject (subject-disjoint): both L and R complete.
    subject_to_hands: Dict[str, set] = defaultdict(set)
    for (subject, hand) in eligible_palmsides:
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
            f"Need {total} eligible subjects but only {len(eligible_subjects)} are complete."
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
            h_flash = huawei_data[(subject, hand)].get("Flash", [])[:10]
            h_nat = huawei_data[(subject, hand)].get("Nature", [])[:10]
            if args.mode == "flash_to_nature":
                enroll = [_rel_to_project(p) for p in h_flash[:5]]
                dev = [_rel_to_project(p) for p in h_flash[5:10]]
                test = [_rel_to_project(p) for p in h_nat[:10]]
            else:
                i_nat = iphone_data[(subject, hand)].get("Nature", [])[:10]
                enroll = [_rel_to_project(p) for p in h_flash[:5]]
                dev = [_rel_to_project(p) for p in h_flash[5:10]]
                test = [_rel_to_project(p) for p in i_nat[:10]]
            sample_split[label] = {"enroll": enroll, "dev": dev, "test": test}
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
        dataset_name=dataset_name,
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
            f"XJTU appendix mode={args.mode}: "
            + ("enroll=Huawei.Flash[:5], dev=Huawei.Flash[5:10], "
               "test=Huawei.Nature[:10]." if args.mode == "flash_to_nature"
               else "enroll=Huawei.Flash[:5], dev=Huawei.Flash[5:10], "
                    "test=iPhone.Nature[:10] (cross-device).")
        ),
        excluded_items=excluded_items,
        seed=args.seed,
        future_subject_order=list(future_ids),
        future_ids_order_source=(
            f"shuffle(sorted(eligible_subjects), seed={args.seed}); "
            "labels in (subject_shuffle_order, hand=R then L)."
        ),
    )

    if not args.no_validate:
        validate_manifest(out_dir, project_root=PROJECT_ROOT)
        print(f"[XJTU] manifest validation passed: {out_dir}")

    write_configs(
        out_dir,
        dataset_name=dataset_name,
        future_batch_size=args.future_batch_size,
        static_gallery_sizes=[30, 60, 100, 130],
        seed=args.seed,
        n_enroll=5, n_dev=5, n_test_min=10,
        n_base=len(base_ids), n_future=len(future_ids), n_external=len(external_ids),
    )

    print(f"[XJTU] mode={args.mode} eligible subjects: {len(eligible_subjects)}")
    print(f"[XJTU] selected: {len(canonical_subject_order)} subjects "
          f"({len(canonical_subject_order)*2} palm-sides) — "
          f"base={len(base_ids)} future={len(future_ids)} external={len(external_ids)}")
    print(f"[XJTU] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
