"""
Phase 2 — Generate 9 variant configs for COCONUT diagnostic tear-down on BJTU.

Reads `config/config.yaml` as the L_full baseline, applies one-line overrides
per variant, and writes each as `config/phase2/<variant>.yaml`.

Variants (per plan §B.4.1, DER++-convention 3-tier structure):
    L_naive     — DER++'s "SGD": NO memory replay, NO CL/openset extras (lower bound)
    L_replay    — DER++'s "ER":  memory replay ON, but all method extras OFF
    L_full      — current config unchanged (proposed method, reference)
    no_proxy    — L_full minus ProxyAnchor
    no_idl      — L_full minus IDL-RTM
    no_der      — L_full minus DER++
    no_qar      — L_full minus QAR
    no_snorm    — L_full minus S-norm
    no_maha     — L_full minus Mahalanobis (cosine score)
    no_recal    — L_full minus τ recalibration (static τ)

L_minimal is generated AFTER Phase 2 leave-one-out results, not here.

Usage:
    python scripts/phase2_setup.py
        --base config/config.yaml \\
        --out_dir config/phase2 \\
        --results_root /content/drive/MyDrive/Secure_V19/exp2_pre

Output:
    config/phase2/L0.yaml
    config/phase2/L_full.yaml
    config/phase2/no_proxy.yaml
    ...
    config/phase2/MANIFEST.json  -- summary of all variants and overrides
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


# Each variant is described as: (description, overrides_dict). Overrides target
# the canonical config.yaml top-level keys (Training / Openset / Model / Dataset).
# A nested key like ("Training", "use_proxy_anchor") means
# cfg["Training"]["use_proxy_anchor"] = value.
VARIANTS = {
    "L_naive": {
        "desc": "DER++'s SGD lower bound — no memory replay, no CL/openset extras",
        "overrides": {
            # Disable memory replay (the defining difference vs L_replay)
            ("Training", "memory_batch_size"): 0,
            # Disable all method components (same as L_replay below)
            ("Training", "use_proxy_anchor"): False,
            ("Training", "use_idl_rtm"): False,
            ("Training", "der_alpha"): 0.0,
            ("Training", "use_qar"): False,
            ("Openset", "use_snorm"): False,
            ("Openset", "score_mode"): "cosine",
            ("Openset", "threshold_alpha"): 1.0,
            ("Openset", "threshold_max_delta"): 1.0,
        },
    },
    "L_replay": {
        "desc": "DER++'s ER baseline — memory replay only, no CL/openset extras",
        "overrides": {
            # Memory replay stays ON (default memory_batch_size=160)
            ("Training", "use_proxy_anchor"): False,
            ("Training", "use_idl_rtm"): False,
            ("Training", "der_alpha"): 0.0,
            ("Training", "use_qar"): False,
            ("Openset", "use_snorm"): False,
            ("Openset", "score_mode"): "cosine",
            ("Openset", "threshold_alpha"): 1.0,
            ("Openset", "threshold_max_delta"): 1.0,
        },
    },
    "L_full": {
        "desc": "Current config — all 7 components on (reference baseline)",
        "overrides": {},  # no changes
    },
    "no_proxy": {
        "desc": "L_full minus ProxyAnchor",
        "overrides": {
            ("Training", "use_proxy_anchor"): False,
        },
    },
    "no_idl": {
        "desc": "L_full minus IDL-RTM",
        "overrides": {
            ("Training", "use_idl_rtm"): False,
        },
    },
    "no_der": {
        "desc": "L_full minus DER++ feature distillation",
        "overrides": {
            ("Training", "der_alpha"): 0.0,
        },
    },
    "no_qar": {
        "desc": "L_full minus QAR tail-user rehab",
        "overrides": {
            ("Training", "use_qar"): False,
        },
    },
    "no_snorm": {
        "desc": "L_full minus S-norm (per-class z-score)",
        "overrides": {
            ("Openset", "use_snorm"): False,
        },
    },
    "no_maha": {
        "desc": "L_full minus Mahalanobis (cosine score)",
        "overrides": {
            ("Openset", "score_mode"): "cosine",
        },
    },
    "no_recal": {
        "desc": "L_full minus τ recalibration (static τ after first calibration)",
        "overrides": {
            ("Openset", "threshold_alpha"): 1.0,
            ("Openset", "threshold_max_delta"): 1.0,
        },
    },
}


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply (top, nested) -> value overrides in-place on a deep copy of cfg."""
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        section, key = path
        out[section][key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base config.yaml (L_full reference)")
    ap.add_argument("--out_dir", required=True, help="output dir for variant configs")
    ap.add_argument("--results_root", required=True,
                    help="root dir for per-variant results (Drive path on Colab)")
    args = ap.parse_args()

    base_path = Path(args.base)
    if not base_path.is_file():
        print(f"ERROR: base config not found: {base_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_path.open() as f:
        base_cfg = yaml.safe_load(f)

    manifest = {
        "base_config": str(base_path),
        "results_root": args.results_root,
        "variants": {},
    }

    for vname, vinfo in VARIANTS.items():
        cfg = apply_overrides(base_cfg, vinfo["overrides"])
        # Per-variant results path so runs do not clobber each other.
        cfg["Training"]["results_path"] = f"{args.results_root}/{vname}"
        cfg["Training"]["checkpoint_path"] = f"{args.results_root}/{vname}"

        out_path = out_dir / f"{vname}.yaml"
        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Snapshot the effective flags for manifest (helps Phase 4 reading).
        effective = {
            "use_proxy_anchor": cfg["Training"].get("use_proxy_anchor"),
            "use_idl_rtm": cfg["Training"].get("use_idl_rtm"),
            "der_alpha": cfg["Training"].get("der_alpha"),
            "use_qar": cfg["Training"].get("use_qar"),
            "use_snorm": cfg["Openset"].get("use_snorm"),
            "score_mode": cfg["Openset"].get("score_mode"),
            "threshold_alpha": cfg["Openset"].get("threshold_alpha"),
            "threshold_max_delta": cfg["Openset"].get("threshold_max_delta"),
        }
        manifest["variants"][vname] = {
            "desc": vinfo["desc"],
            "config": str(out_path),
            "results_path": cfg["Training"]["results_path"],
            "overrides": {f"{p[0]}.{p[1]}": v for p, v in vinfo["overrides"].items()},
            "effective_flags": effective,
        }
        print(f"  {vname:10s} -> {out_path}")

    manifest_path = out_dir / "MANIFEST.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
