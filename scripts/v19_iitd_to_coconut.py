"""
V19 IITD manifest -> COCONUT 5-txt converter.

Reads V19 canonical manifest.csv and writes 5 txt files expected by
the COCONUT data layer (path + space + integer label per line):

  enroll.txt        gallery candidates for sequential enrollment
                    (subject_split in {base_anchor, future}, per-palm 80% split)
  eval_probe.txt    held-out probe samples from same gallery palms (per-palm 20%)
  unknown_dev.txt   external_dev split  (FAR cohort for calibration)
  unknown_test.txt  external_test split (FAR cohort for final measurement)
  xdomain.txt       cross-domain reference (BJTU train_left.txt copied verbatim)
  label_map.json    palm_id -> dense gallery label remap (paper-repro)

Per-palm split: samples sorted by (sample_id ascending), ceil(n * enroll_frac)
go to enroll, remainder to eval_probe. Deterministic given fixed manifest order.

Path remapping: --path_remap "OLD=NEW" rewrites image_path prefix for runtime
environment (e.g. local mac -> Colab Drive). Multiple --path_remap allowed.
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

GALLERY_SPLITS = ("base_anchor", "future")
UNKNOWN_DEV_SPLIT = "external_dev"
UNKNOWN_TEST_SPLIT = "external_test"
UNKNOWN_LABEL_OFFSET = 10000  # keep unknown labels disjoint from gallery dense [0..N-1]


def apply_path_remap(path: str, remaps: list[tuple[str, str]]) -> str:
    for old, new in remaps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def write_txt(out_path: Path, rows: list[tuple[str, int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p, lab in rows:
            f.write(f"{p} {lab}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="path to V19 iitd_full/manifest.csv")
    ap.add_argument("--out_dir", required=True, help="output dir for 5 txt + label_map.json")
    ap.add_argument("--enroll_frac", type=float, default=0.8,
                    help="per-palm fraction sent to enroll.txt; rest -> eval_probe.txt (default 0.8)")
    ap.add_argument("--xdomain_src", default=None,
                    help="path to BJTU train_left.txt to copy as xdomain.txt (optional)")
    ap.add_argument("--path_remap", action="append", default=[],
                    help='prefix rewrite, format "OLD=NEW" (repeatable)')
    args = ap.parse_args()

    remaps: list[tuple[str, str]] = []
    for spec in args.path_remap:
        if "=" not in spec:
            print(f"ERROR: --path_remap must be 'OLD=NEW', got: {spec}", file=sys.stderr)
            return 2
        old, new = spec.split("=", 1)
        remaps.append((old, new))

    df = pd.read_csv(args.manifest)
    required_cols = {"image_path", "palm_id", "subject_id", "sample_id", "subject_split"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: manifest missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Gallery (base_anchor + future) -----
    gallery = df[df["subject_split"].isin(GALLERY_SPLITS)].copy()
    gallery_palms = sorted(gallery["palm_id"].unique().tolist())
    label_map = {int(pid): i for i, pid in enumerate(gallery_palms)}

    enroll_rows: list[tuple[str, int]] = []
    probe_rows: list[tuple[str, int]] = []
    per_palm_counts: list[dict] = []

    for pid in gallery_palms:
        palm_df = gallery[gallery["palm_id"] == pid].sort_values("sample_id")
        n = len(palm_df)
        n_enroll = max(1, math.ceil(n * args.enroll_frac))
        n_enroll = min(n_enroll, n - 1) if n > 1 else n  # always leave >=1 for probe if n>1
        dense_label = label_map[int(pid)]

        for j, row in enumerate(palm_df.itertuples(index=False)):
            path = apply_path_remap(str(row.image_path), remaps)
            if j < n_enroll:
                enroll_rows.append((path, dense_label))
            else:
                probe_rows.append((path, dense_label))
        per_palm_counts.append(dict(palm_id=int(pid), dense_label=dense_label, n=n,
                                    n_enroll=n_enroll, n_probe=n - n_enroll))

    # ----- Unknown cohorts -----
    def unknown_rows(split_name: str) -> list[tuple[str, int]]:
        sub = df[df["subject_split"] == split_name].sort_values(["palm_id", "sample_id"])
        rows = []
        for row in sub.itertuples(index=False):
            path = apply_path_remap(str(row.image_path), remaps)
            rows.append((path, int(row.palm_id) + UNKNOWN_LABEL_OFFSET))
        return rows

    unk_dev = unknown_rows(UNKNOWN_DEV_SPLIT)
    unk_test = unknown_rows(UNKNOWN_TEST_SPLIT)

    write_txt(out_dir / "enroll.txt", enroll_rows)
    write_txt(out_dir / "eval_probe.txt", probe_rows)
    write_txt(out_dir / "unknown_dev.txt", unk_dev)
    write_txt(out_dir / "unknown_test.txt", unk_test)

    # ----- xdomain (optional copy from BJTU) -----
    xdom_path = out_dir / "xdomain.txt"
    if args.xdomain_src:
        src = Path(args.xdomain_src)
        if not src.is_file():
            print(f"WARN: --xdomain_src not found: {src} (skipping xdomain.txt)", file=sys.stderr)
        else:
            if remaps:
                # Apply path_remap to xdomain content too.
                with src.open() as fr, xdom_path.open("w") as fw:
                    for line in fr:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        parts = line.rsplit(" ", 1)
                        if len(parts) != 2:
                            continue
                        p, lab = parts
                        p = apply_path_remap(p, remaps)
                        fw.write(f"{p} {lab}\n")
            else:
                shutil.copy(src, xdom_path)

    with (out_dir / "label_map.json").open("w") as f:
        json.dump({
            "gallery_palm_count": len(gallery_palms),
            "enroll_frac": args.enroll_frac,
            "unknown_label_offset": UNKNOWN_LABEL_OFFSET,
            "palm_id_to_dense_label": label_map,
            "per_palm_counts": per_palm_counts,
            "n_enroll_total": len(enroll_rows),
            "n_probe_total": len(probe_rows),
            "n_unknown_dev": len(unk_dev),
            "n_unknown_test": len(unk_test),
        }, f, indent=2)

    print(f"OK -> {out_dir}")
    print(f"  enroll.txt       {len(enroll_rows):5d} lines  ({len(gallery_palms)} palms)")
    print(f"  eval_probe.txt   {len(probe_rows):5d} lines")
    print(f"  unknown_dev.txt  {len(unk_dev):5d} lines")
    print(f"  unknown_test.txt {len(unk_test):5d} lines")
    print(f"  xdomain.txt      {'written' if xdom_path.exists() else 'skipped'}")
    print(f"  label_map.json   {len(label_map)} palm_id -> dense labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
