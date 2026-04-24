"""
Experiment 1: Protocol Comparison — 시각화 (v5)

Fig 1: Gallery-size effect + size-matched sanity (External FPIR main axis)
Fig 2: Threshold strategy (C-fixed / C-recalib / C-snorm-fixed curves) — 3 subplot
Fig 3: Score distribution (step 0 vs final, raw cosine + S-norm panel)
Appendix: Not-yet-enrolled rejection (sample-size annotated)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List
from pathlib import Path


COLORS = {
    'B':           '#666666',   # static bars
    'C_raw_fixed':     '#4A90D9',
    'C_raw_recalib':   '#2E8B57',
    'C_snorm_fixed':   '#E8943A',
    'C_snorm_recalib': '#A35EDB',
}


def _setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ============================================================
# Fig 1: Gallery-size effect + size-matched sanity
# ============================================================

def plot_fig1_gallery_size(b_results_by_size: Dict[int, Dict],
                           c_raw_fixed: Dict,
                           c_raw_recalib: Dict,
                           save_path: str,
                           dataset_label: str = ''):
    """
    2 subplot (TPIR, External FPIR).
    x: gallery size. Static bars + C-fixed-final / C-recalib-final overlay.
    """
    _setup_style()

    sizes = sorted(b_results_by_size.keys())
    tpir_vals = [b_results_by_size[s]['tpir_at_1pct_fpir'] for s in sizes]
    ext_fpir_vals = [b_results_by_size[s]['achieved_external_fpir'] for s in sizes]
    ext_rej_vals = [b_results_by_size[s]['external_rejection_rate'] for s in sizes]

    c_final_rf = c_raw_fixed['steps'][-1]
    c_final_rr = c_raw_recalib['steps'][-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: TPIR@1%FPIR
    ax = axes[0]
    bars = ax.bar(range(len(sizes)), tpir_vals, color=COLORS['B'], alpha=0.75,
                  edgecolor='black', linewidth=0.5, width=0.6, label='Static B-s')
    for xi, val in enumerate(tpir_vals):
        ax.text(xi, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    # Overlay C-fixed-final / C-recalib-final at the last size
    final_x = len(sizes) - 1
    ax.scatter([final_x - 0.15], [c_final_rf['tpir_at_1pct_fpir']],
               color=COLORS['C_raw_fixed'], s=80, zorder=5, marker='o',
               label=f"C-raw-fixed final ({c_final_rf['tpir_at_1pct_fpir']:.3f})")
    ax.scatter([final_x + 0.15], [c_final_rr['tpir_at_1pct_fpir']],
               color=COLORS['C_raw_recalib'], s=80, zorder=5, marker='s',
               label=f"C-raw-recalib final ({c_final_rr['tpir_at_1pct_fpir']:.3f})")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('Gallery size')
    ax.set_ylabel('TPIR at dev-calibrated 1% FPIR threshold')
    ax.set_title('TPIR vs gallery size')
    ax.set_ylim(0, min(1.05, max(tpir_vals + [c_final_rf['tpir_at_1pct_fpir'],
                                               c_final_rr['tpir_at_1pct_fpir']]) + 0.12))
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: External FPIR (main axis) + Rejection Rate secondary
    ax = axes[1]
    bars2 = ax.bar(range(len(sizes)), ext_fpir_vals, color=COLORS['B'], alpha=0.75,
                   edgecolor='black', linewidth=0.5, width=0.6, label='Static B-s')
    for xi, val in enumerate(ext_fpir_vals):
        ax.text(xi, val + max(ext_fpir_vals) * 0.02, f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)

    final_x = len(sizes) - 1
    ax.scatter([final_x - 0.15], [c_final_rf['achieved_external_fpir']],
               color=COLORS['C_raw_fixed'], s=80, zorder=5, marker='o',
               label=f"C-raw-fixed final ({c_final_rf['achieved_external_fpir']:.4f})")
    ax.scatter([final_x + 0.15], [c_final_rr['achieved_external_fpir']],
               color=COLORS['C_raw_recalib'], s=80, zorder=5, marker='s',
               label=f"C-raw-recalib final ({c_final_rr['achieved_external_fpir']:.4f})")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('Gallery size')
    ax.set_ylabel('Achieved External FPIR (test)')
    ax.set_title('False accept risk vs gallery size')
    ymax = max(ext_fpir_vals + [c_final_rf['achieved_external_fpir'],
                                 c_final_rr['achieved_external_fpir']])
    ax.set_ylim(0, ymax * 1.3 + 1e-6)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # secondary axis: rejection rate
    ax2 = ax.twinx()
    ax2.plot(range(len(sizes)), ext_rej_vals, color='gray', linestyle=':', marker='x',
             alpha=0.6, label='Rejection Rate (= 1−FPIR)')
    ax2.set_ylabel('External Rejection Rate', color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.set_ylim(min(ext_rej_vals) - 0.02, 1.005)

    title = 'Fig 1: Gallery-size sweep + size-matched sanity'
    if dataset_label:
        title += f'  [{dataset_label}]'
    fig.suptitle(title, fontsize=13, y=1.02)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 1] saved: {save_path}")


# ============================================================
# Fig 2: Threshold strategy curves
# ============================================================

def plot_fig2_sequential_curves(c_raw_fixed: Dict,
                                c_raw_recalib: Dict,
                                c_snorm_fixed: Dict,
                                c_snorm_recalib: Dict,
                                save_path: str):
    """
    3 subplot:
      (a) TPIR@dev-calibrated 1% FPIR
      (b) Achieved External FPIR (test)
      (c) Threshold trajectory — raw / S-norm subplot 분리
    """
    _setup_style()

    conditions = [
        ('C_raw_fixed', c_raw_fixed, 'C-raw-fixed', 'o-'),
        ('C_raw_recalib', c_raw_recalib, 'C-raw-recalib', 's--'),
        ('C_snorm_fixed', c_snorm_fixed, 'C-snorm-fixed', '^-'),
        ('C_snorm_recalib', c_snorm_recalib, 'C-snorm-recalib (app.)', 'd:'),
    ]

    # x-axis: gallery size
    xs = {}
    for key, res, _, _ in conditions:
        xs[key] = [step['gallery_size'] for step in res['steps']]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    # (a) TPIR
    ax = axes[0]
    for key, res, label, style in conditions:
        vals = [s['tpir_at_1pct_fpir'] for s in res['steps']]
        ax.plot(xs[key], vals, style, color=COLORS[key], label=label,
                linewidth=1.8, markersize=5)
    ax.set_ylabel('TPIR at dev-calibrated 1% FPIR')
    ax.set_title('(a) TPIR vs gallery size')
    ax.set_xlabel('Enrolled identities')
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Achieved External FPIR (test)
    ax = axes[1]
    for key, res, label, style in conditions:
        vals = [s['achieved_external_fpir'] for s in res['steps']]
        ax.plot(xs[key], vals, style, color=COLORS[key], label=label,
                linewidth=1.8, markersize=5)
    ax.set_ylabel('Achieved External FPIR (test)')
    ax.set_title('(b) External false-accept risk vs gallery size')
    ax.set_xlabel('Enrolled identities')
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Threshold trajectory — raw / S-norm 분리 (twin y-axis)
    ax = axes[2]
    # Raw: C_raw_fixed, C_raw_recalib
    for key, res, label, style in [
        ('C_raw_fixed', c_raw_fixed, 'C-raw-fixed (raw)', 'o-'),
        ('C_raw_recalib', c_raw_recalib, 'C-raw-recalib (raw)', 's--'),
    ]:
        vals = [s['threshold'] for s in res['steps']]
        ax.plot(xs[key], vals, style, color=COLORS[key], label=label,
                linewidth=1.8, markersize=5)
    ax.set_ylabel('Raw cosine threshold', color='#333333')
    ax.set_xlabel('Enrolled identities')
    ax.set_title('(c) Threshold trajectory — raw vs S-norm (separate axes)')

    ax_sn = ax.twinx()
    for key, res, label, style in [
        ('C_snorm_fixed', c_snorm_fixed, 'C-snorm-fixed (S-norm)', '^-'),
        ('C_snorm_recalib', c_snorm_recalib, 'C-snorm-recalib (S-norm)', 'd:'),
    ]:
        vals = [s['threshold'] for s in res['steps']]
        ax_sn.plot(xs[key], vals, style, color=COLORS[key], label=label,
                   linewidth=1.8, markersize=5, alpha=0.85)
    ax_sn.set_ylabel('S-norm threshold (z-score)', color='#884400')
    ax_sn.tick_params(axis='y', colors='#884400')

    # combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_sn.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)

    fig.suptitle('Fig 2: Sequential threshold strategies (C variants)',
                 fontsize=13, y=1.00)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 2] saved: {save_path}")


# ============================================================
# Fig 3: Score distribution (step 0 vs final) + S-norm panel
# ============================================================

def plot_fig3_score_distributions(c_raw_fixed: Dict,
                                  c_raw_recalib: Dict,
                                  c_snorm_fixed: Dict,
                                  save_path: str):
    """
    3 panel:
      (a) Raw cosine distribution step 0 + C-fixed threshold line (step 0)
                                       + C-recalib threshold line (step 0, 같음)
      (b) Raw cosine distribution final + C-fixed threshold (step 0, 고정)
                                        + C-recalib threshold (final)
      (c) S-norm distribution step 0 vs final (small panel)

    Raw: C-fixed와 C-recalib의 score distribution은 동일
         (같은 gallery + 같은 probe + 같은 raw cosine). threshold만 overlay 다름.
    """
    _setup_style()

    sd_rf = c_raw_fixed.get('score_distributions', {})
    sd_rr = c_raw_recalib.get('score_distributions', {})
    sd_sf = c_snorm_fixed.get('score_distributions', {})

    if not sd_rf or not sd_rr:
        print("[FIGURE 3] score_distributions missing, skipping")
        return

    rf_steps = c_raw_fixed['steps']
    rr_steps = c_raw_recalib['steps']
    sf_steps = c_snorm_fixed['steps']

    step0_key = rf_steps[0]['step']
    final_key = rf_steps[-1]['step']

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
    ax0 = fig.add_subplot(gs[0, 0])   # raw step 0
    axF = fig.add_subplot(gs[0, 1])   # raw final
    axS0 = fig.add_subplot(gs[1, 0])  # snorm step 0
    axSF = fig.add_subplot(gs[1, 1])  # snorm final
    axLegend = fig.add_subplot(gs[:, 2])   # legend/info

    bins = np.linspace(-0.2, 1.0, 60)
    bins_snorm = 60

    def _plot_raw_panel(ax, sd_step, title, th_fixed, th_recalib):
        if sd_step.get('known'):
            ax.hist(sd_step['known'], bins=bins, alpha=0.55, color='#2E8B57',
                    label=f"Known (n={len(sd_step['known'])})", density=True)
        if sd_step.get('future'):
            ax.hist(sd_step['future'], bins=bins, alpha=0.45, color='#DC3545',
                    label=f"Not-yet-enrolled (n={len(sd_step['future'])})", density=True)
        if sd_step.get('external'):
            ax.hist(sd_step['external'], bins=bins, alpha=0.45, color='#4A90D9',
                    label=f"External (n={len(sd_step['external'])})", density=True)
        if th_fixed is not None:
            ax.axvline(th_fixed, color=COLORS['C_raw_fixed'], linestyle='--',
                       linewidth=1.8, label=f"C-raw-fixed τ={th_fixed:.3f}")
        if th_recalib is not None:
            ax.axvline(th_recalib, color=COLORS['C_raw_recalib'], linestyle=':',
                       linewidth=1.8, label=f"C-raw-recalib τ={th_recalib:.3f}")
        ax.set_xlabel('Max cosine score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Raw step 0: C-fixed τ = step 0, C-recalib τ = step 0 (same)
    _plot_raw_panel(
        ax0, sd_rf.get(step0_key, {}),
        f'(a) Raw cosine @ {step0_key}',
        rf_steps[0]['threshold'],
        rr_steps[0]['threshold'],
    )
    ax0.legend(fontsize=7, loc='upper left')

    # Raw final: C-fixed τ = step 0 (carried over), C-recalib τ = final
    _plot_raw_panel(
        axF, sd_rf.get(final_key, {}),
        f'(b) Raw cosine @ {final_key}',
        rf_steps[-1]['threshold'],
        rr_steps[-1]['threshold'],
    )
    axF.legend(fontsize=7, loc='upper left')

    # S-norm step 0
    def _plot_snorm_panel(ax, sd_step, title, th):
        if sd_step.get('known'):
            ax.hist(sd_step['known'], bins=bins_snorm, alpha=0.55, color='#2E8B57',
                    label=f"Known (n={len(sd_step['known'])})", density=True)
        if sd_step.get('external'):
            ax.hist(sd_step['external'], bins=bins_snorm, alpha=0.45, color='#4A90D9',
                    label=f"External (n={len(sd_step['external'])})", density=True)
        if th is not None:
            ax.axvline(th, color=COLORS['C_snorm_fixed'], linestyle='--',
                       linewidth=1.8, label=f"C-snorm-fixed τ={th:.3f}")
        ax.set_xlabel('Max S-norm score (z)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if sd_sf:
        _plot_snorm_panel(
            axS0, sd_sf.get(step0_key, {}),
            f'(c) S-norm @ {step0_key}',
            sf_steps[0]['threshold'],
        )
        axS0.legend(fontsize=7, loc='upper left')
        _plot_snorm_panel(
            axSF, sd_sf.get(final_key, {}),
            f'(d) S-norm @ {final_key}',
            sf_steps[-1]['threshold'],   # fixed: same as step 0 threshold
        )
        axSF.legend(fontsize=7, loc='upper left')
    else:
        axS0.text(0.5, 0.5, 'no S-norm data', ha='center', va='center')
        axSF.text(0.5, 0.5, 'no S-norm data', ha='center', va='center')

    # Legend/info panel
    axLegend.axis('off')
    info = (
        "Fig 3: Score distributions\n"
        "\n"
        "Top row (a,b):\n"
        "  Raw cosine scores.\n"
        "  C-raw-fixed and C-raw-recalib share\n"
        "  the SAME raw score distribution\n"
        "  — only their thresholds differ.\n"
        "\n"
        "Bottom row (c,d):\n"
        "  S-norm z-scores.\n"
        "  Different score space from raw.\n"
        "  C-snorm-fixed uses step-0 threshold\n"
        "  in S-norm space throughout."
    )
    axLegend.text(0.0, 0.95, info, va='top', fontsize=9, family='monospace')

    fig.suptitle('Fig 3: Score distributions with threshold overlays',
                 fontsize=13, y=1.00)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 3] saved: {save_path}")


# ============================================================
# Appendix: Not-yet-enrolled rejection
# ============================================================

def plot_appendix_nye_rejection(c_raw_fixed: Dict,
                                c_raw_recalib: Dict,
                                save_path: str,
                                min_probes_for_solid: int = 50):
    """
    Not-yet-enrolled rejection curve. 후반 step의 sample size가 너무 작으면
    회색 또는 점선으로 표시.
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    for res, label, color, base_style in [
        (c_raw_fixed, 'C-raw-fixed', COLORS['C_raw_fixed'], 'o-'),
        (c_raw_recalib, 'C-raw-recalib', COLORS['C_raw_recalib'], 's--'),
    ]:
        xs = [s['gallery_size'] for s in res['steps']]
        vals = [s['not_yet_enrolled_rejection'] for s in res['steps']]
        probe_counts = [s['not_yet_enrolled_probes'] for s in res['steps']]

        # solid segment: probes >= threshold
        solid_mask = [n >= min_probes_for_solid and n > 0 for n in probe_counts]
        solid_x = [x for x, m in zip(xs, solid_mask) if m]
        solid_y = [y for y, m in zip(vals, solid_mask) if m]
        faded_x = [x for x, m in zip(xs, solid_mask) if not m]
        faded_y = [y for y, m in zip(vals, solid_mask) if not m]

        ax.plot(solid_x, solid_y, base_style, color=color, label=label,
                linewidth=2, markersize=6)
        if faded_x:
            ax.plot(faded_x, faded_y, 'x', color=color, alpha=0.4,
                    label=f'{label} (n<{min_probes_for_solid}, dashed)', markersize=7)
            # draw dashed connector to show trajectory
            combined = sorted(zip(xs, vals))
            cx = [c[0] for c in combined]
            cy = [c[1] for c in combined]
            ax.plot(cx, cy, color=color, linestyle=':', alpha=0.3, linewidth=1)

        # annotate probe count on a couple of faded points
        for x, y, n in zip(xs, vals, probe_counts):
            if n < min_probes_for_solid:
                ax.annotate(f'n={n}', (x, y), textcoords='offset points',
                            xytext=(5, -10), fontsize=7, color='gray')

    ax.set_xlabel('Enrolled identities')
    ax.set_ylabel('Not-yet-enrolled rejection rate')
    ax.set_title('Appendix: Not-yet-enrolled rejection\n'
                 f'(points with probe count < {min_probes_for_solid} shown faded)')
    ax.legend(fontsize=9, loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[APPENDIX FIGURE] saved: {save_path}")


# ============================================================
# Console summary
# ============================================================

def plot_console_summary(all_condition_results: Dict, sanity_summary: Dict):
    """Final-step summary table 콘솔 출력."""
    print("\n" + "=" * 90)
    print("  EXPERIMENT 1 SUMMARY — Final-step metrics")
    print("=" * 90)
    print(f"  {'Condition':<22} {'Gallery':<10} {'Rank-1':<10} "
          f"{'TPIR@1%':<10} {'ExtRej':<10} {'AchFPIR':<10} {'τ':<10}")
    print("-" * 90)

    def _row(label, step, tau_field='threshold'):
        print(f"  {label:<22} {step['gallery_size']:<10} "
              f"{step['rank1']:.4f}    "
              f"{step['tpir_at_1pct_fpir']:.4f}    "
              f"{step['external_rejection_rate']:.4f}    "
              f"{step['achieved_external_fpir']:.4f}    "
              f"{step.get(tau_field, float('nan')):.4f}")

    # A
    a = all_condition_results['A']
    a_final = a['steps'][-1]
    print(f"  {'A (closed-set)':<22} {a_final['gallery_size']:<10} "
          f"{a_final['rank1']:.4f}    {'—':<10} {'—':<10} {'—':<10} {'—':<10}")

    # B-s
    for size, res in sorted(all_condition_results['B'].items()):
        # res is not a step — it's a static snapshot
        print(f"  {'B-'+str(size):<22} {res['gallery_size']:<10} "
              f"{res['rank1']:.4f}    "
              f"{res['tpir_at_1pct_fpir']:.4f}    "
              f"{res['external_rejection_rate']:.4f}    "
              f"{res['achieved_external_fpir']:.4f}    "
              f"{res['threshold']:.4f}")

    # C variants
    for key, label in [
        ('C_raw_fixed', 'C-raw-fixed'),
        ('C_raw_recalib', 'C-raw-recalib'),
        ('C_snorm_fixed', 'C-snorm-fixed'),
        ('C_snorm_recalib', 'C-snorm-recalib'),
    ]:
        res = all_condition_results[key]
        _row(label + ' (final)', res['steps'][-1])

    print("=" * 90)
    print("\n[SANITY CHECK — B-sF vs C-raw-recalib-final]")
    for k, v in sanity_summary.items():
        if isinstance(v, bool):
            mark = '✓' if v else '✗'
            print(f"  {k:30s} = {v}   {mark}")
        elif isinstance(v, float):
            print(f"  {k:30s} = {v:.6e}")
        else:
            print(f"  {k:30s} = {v}")
    print()
