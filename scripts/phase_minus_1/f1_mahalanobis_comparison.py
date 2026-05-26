"""
Phase -1.1: F1 — accurate Mahalanobis impact comparison.

Compares 4 variants at N=50 (trimming L_full from N=100 → N=50):
  - L_full              (Maha + DER + IDL + Replay + Proxy + S-norm + QAR)
  - no_maha             (L_full minus Mahalanobis)
  - only_maha           (Maha alone, no Replay/QAR)
  - L_minimal_no_supcon (5-component removal series; no Maha)

Goal: directly verify whether removing Mahalanobis as a SCORE MODE
(replacing cosine with Mahalanobis distance) is helpful, harmful, or
redundant — under apples-to-apples comparison.

Note: this does NOT test "GHOST-style inference normalization gate".
That requires a separate experiment (Phase -1.6).

Compute: 0 — reads existing Drive CSV files.

Usage (Colab):
    python scripts/phase_minus_1/f1_mahalanobis_comparison.py \
        --base /content/drive/MyDrive/Secure_V19/exp2_pre \
        --output /content/drive/MyDrive/phase_minus_1/f1_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


VARIANTS = ['L_full', 'no_maha', 'only_maha', 'L_minimal_no_supcon']
LATE_WINDOW = (40, 50)  # inclusive


def load_drift(variant_dir: str) -> Optional[pd.DataFrame]:
    p = Path(variant_dir) / 'coconut_results' / 'fpir_drift_log.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


def summarize_at_n(df: pd.DataFrame, n_target: int = 50, window=(40, 50)) -> Dict:
    """Summarize variant at N=n_target with late-stage window stats."""
    if df is None or df.empty:
        return {'error': 'no data'}

    # Trim to N=n_target if longer
    df_trim = df[df['experience'] <= n_target].copy()
    final_n = int(df_trim['experience'].max())
    if final_n < n_target:
        print(f"  ⚠️  variant only runs to N={final_n}, not {n_target}")

    final_row = df_trim[df_trim['experience'] == final_n].iloc[0]
    late = df_trim[
        (df_trim['experience'] >= window[0]) & (df_trim['experience'] <= min(window[1], final_n))
    ]

    out = {
        'final_n': final_n,
        'n_exp_total': int(len(df)),
        'FinalTAR': float(final_row['avg_tar_001']),
        'window': f"{window[0]}-{min(window[1], final_n)}",
        'MeanTAR_late': float(late['avg_tar_001'].mean()) if len(late) > 0 else None,
        'StdTAR_late': float(late['avg_tar_001'].std()) if len(late) > 0 else None,
        'MinTAR_late': float(late['avg_tar_001'].min()) if len(late) > 0 else None,
        'MaxTAR_late': float(late['avg_tar_001'].max()) if len(late) > 0 else None,
        'Final_BWT': float(final_row['bwt']),
        'Final_Forgetting': float(final_row['forgetting']),
        'Final_FPIR_in': float(final_row.get('fpir_in', np.nan)),
        'Final_tau': float(final_row.get('tau', np.nan)),
    }
    return out


def compute_deltas(results: Dict[str, Dict]) -> Dict:
    """Compute key paired deltas between variants."""
    deltas = {}

    if 'L_full' in results and 'no_maha' in results:
        lf = results['L_full']
        nm = results['no_maha']
        if 'error' not in lf and 'error' not in nm:
            deltas['L_full_minus_no_maha'] = {
                'FinalTAR_delta': lf['FinalTAR'] - nm['FinalTAR'],
                'MeanTAR_delta': lf['MeanTAR_late'] - nm['MeanTAR_late']
                if nm['MeanTAR_late'] is not None and lf['MeanTAR_late'] is not None else None,
                'BWT_delta': lf['Final_BWT'] - nm['Final_BWT'],
                'forget_delta': lf['Final_Forgetting'] - nm['Final_Forgetting'],
                'description': 'Adding Mahalanobis as score mode on top of all other components',
            }

    if 'L_minimal_no_supcon' in results and 'no_maha' in results:
        lm = results['L_minimal_no_supcon']
        nm = results['no_maha']
        if 'error' not in lm and 'error' not in nm:
            deltas['no_maha_minus_L_minimal_no_supcon'] = {
                'MeanTAR_delta': nm['MeanTAR_late'] - lm['MeanTAR_late']
                if nm['MeanTAR_late'] is not None and lm['MeanTAR_late'] is not None else None,
                'BWT_delta': nm['Final_BWT'] - lm['Final_BWT'],
                'description': 'Both have no Mahalanobis. Are they equivalent?',
            }

    if 'only_maha' in results and 'L_minimal_no_supcon' in results:
        om = results['only_maha']
        lm = results['L_minimal_no_supcon']
        if 'error' not in om and 'error' not in lm:
            deltas['only_maha_minus_L_minimal_no_supcon'] = {
                'MeanTAR_delta': om['MeanTAR_late'] - lm['MeanTAR_late']
                if om['MeanTAR_late'] is not None and lm['MeanTAR_late'] is not None else None,
                'BWT_delta': om['Final_BWT'] - lm['Final_BWT'],
                'description': 'Mahalanobis ALONE vs 5-component minimum without it',
            }

    return deltas


def render_verdict(results: Dict, deltas: Dict) -> str:
    lines = ["=== F1 Verdict ===\n"]

    lf_nm = deltas.get('L_full_minus_no_maha', {})
    if 'MeanTAR_delta' in lf_nm and lf_nm['MeanTAR_delta'] is not None:
        d = lf_nm['MeanTAR_delta']
        if abs(d) < 0.005:
            v = "≈0 → Mahalanobis REDUNDANT (no harm, no benefit)"
        elif d > 0:
            v = f"+{d:.4f} → Mahalanobis HELPFUL (under L_full bundle)"
        else:
            v = f"{d:.4f} → Mahalanobis HARMFUL (under L_full bundle)"
        lines.append(f"L_full vs no_maha (MeanTAR_late at N=50): {v}")

    om_lm = deltas.get('only_maha_minus_L_minimal_no_supcon', {})
    if 'MeanTAR_delta' in om_lm and om_lm['MeanTAR_delta'] is not None:
        d = om_lm['MeanTAR_delta']
        if d < -0.1:
            v = f"{d:.4f} → Mahalanobis alone INSUFFICIENT (needs Replay+QAR)"
        else:
            v = f"{d:.4f} → Mahalanobis alone is comparable"
        lines.append(f"only_maha vs L_minimal_no_supcon (MeanTAR): {v}")

    lines.append("\nGPT 'Gaussian gate failed' claim:")
    lines.append("  - Mahalanobis as score mode was tested (above)")
    lines.append("  - 'Gaussian inference gate as additional rejection' was NOT tested separately")
    lines.append("  - Phase -1.6 (training-free GHOST normalization) will test that claim directly")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True,
                        help='Base path containing variant subdirectories')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f"[1.1] base path: {args.base}")
    print(f"[1.1] target N for trim: 50")
    print(f"[1.1] late window: {LATE_WINDOW[0]}-{LATE_WINDOW[1]}")

    results = {}
    for variant in VARIANTS:
        variant_dir = os.path.join(args.base, variant)
        df = load_drift(variant_dir)
        if df is None:
            print(f"  [{variant}] NO DATA")
            results[variant] = {'error': 'no fpir_drift_log.csv found'}
            continue
        results[variant] = summarize_at_n(df, n_target=50, window=LATE_WINDOW)
        r = results[variant]
        print(f"  [{variant}] N={r.get('final_n')}, "
              f"FinalTAR={r.get('FinalTAR'):.4f}, "
              f"MeanTAR_late={r.get('MeanTAR_late')}, "
              f"BWT={r.get('Final_BWT'):.4f}")

    deltas = compute_deltas(results)
    verdict = render_verdict(results, deltas)
    print("\n" + verdict)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'variants': results,
            'deltas': deltas,
            'verdict_text': verdict,
        }, f, indent=2, default=float)
    print(f"\n[1.1] saved: {args.output}")


if __name__ == '__main__':
    main()
