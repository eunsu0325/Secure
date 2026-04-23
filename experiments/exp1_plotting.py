"""
Experiment 1: Protocol Comparison — 시각화
Figure 1: Summary Table + Bar Chart
Figure 2: Sequential Enrollment Curve (메인 그림)
Figure 3: Score Distribution (max cosine score, step 0 vs final)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버/CLI 환경용
from typing import Dict, List, Optional
from pathlib import Path


def _setup_style():
    """Publication-quality 기본 스타일"""
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
# Figure 1: Summary Table + Rank-1 비교 Bar Chart
# ============================================================

def plot_summary_table(results_a: Dict, results_b: Dict, results_c: Dict,
                       save_path: str):
    """
    세 프로토콜 결과 요약 테이블 + 핵심 Rank-1 비교 bar chart.
    """
    _setup_style()

    # 최종 step 값 추출
    a_final = results_a['steps'][-1]
    c_final = results_c['steps'][-1]

    # --- 콘솔 출력 ---
    print("\n" + "=" * 70)
    print("  PROTOCOL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Protocol':<30} {'Metric':<25} {'Value':<10}")
    print("-" * 70)
    print(f"  {'A: Closed-set CIL':<30} {'Rank-1':<25} {a_final['rank1']:.4f}")
    print(f"  {'B: Static Open-set':<30} {'Known Rank-1':<25} {results_b['known_rank1']:.4f}")
    print(f"  {'B: Static Open-set':<30} {'Known Acceptance':<25} {results_b['known_acceptance']:.4f}")
    print(f"  {'B: Static Open-set':<30} {'External Rejection':<25} {results_b['external_rejection_rate']:.4f}")
    print(f"  {'C: Sequential Enrollment':<30} {'Known Rank-1':<25} {c_final['known_rank1']:.4f}")
    print(f"  {'C: Sequential Enrollment':<30} {'Known Acceptance':<25} {c_final['known_acceptance']:.4f}")
    print(f"  {'C: Sequential Enrollment':<30} {'Future Rejection':<25} {c_final.get('future_rejection', 'N/A')}")
    print(f"  {'C: Sequential Enrollment':<30} {'External Rejection':<25} {c_final['external_rejection']:.4f}")
    print("=" * 70)

    # --- Bar Chart: Rank-1 비교 ---
    fig, ax = plt.subplots(figsize=(6, 4))

    protocols = ['Protocol A\n(Closed-set)', 'Protocol B\n(Static Open-set)',
                 'Protocol C\n(Seq. Enrollment)']
    rank1_values = [a_final['rank1'], results_b['known_rank1'], c_final['known_rank1']]
    colors = ['#4A90D9', '#E8943A', '#50B060']

    bars = ax.bar(protocols, rank1_values, color=colors, width=0.5, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, rank1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Rank-1 Accuracy')
    ax.set_title('Rank-1 under Different Evaluation Protocols')
    ax.set_ylim(0, min(1.05, max(rank1_values) + 0.08))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 1] saved: {save_path}")


# ============================================================
# Figure 2: Sequential Enrollment Curve (메인 그림)
# ============================================================

def plot_sequential_curves(results_c: Dict, save_path: str):
    """
    Protocol C step별 변화 곡선.
    x: enrolled IDs 수
    y: known Rank-1, known acceptance, future rejection, external rejection
    """
    _setup_style()

    steps = results_c['steps']
    n_enrolled = [s['n_enrolled'] for s in steps]
    known_rank1 = [s['known_rank1'] for s in steps]
    known_accept = [s['known_acceptance'] for s in steps]
    future_rej = [s['future_rejection'] for s in steps]
    external_rej = [s['external_rejection'] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n_enrolled, known_rank1, 'o-', color='#2E8B57', linewidth=2,
            markersize=5, label='Known Rank-1')
    ax.plot(n_enrolled, known_accept, 's--', color='#2E8B57', linewidth=1.5,
            markersize=4, alpha=0.7, label='Known Acceptance')

    # future rejection: 마지막 step에서는 future가 없으므로 그때까지만
    future_steps = [(n, r) for n, r, s in zip(n_enrolled, future_rej, steps)
                    if s['future_probes'] > 0]
    if future_steps:
        fn, fr = zip(*future_steps)
        ax.plot(fn, fr, '^-', color='#DC3545', linewidth=2,
                markersize=5, label='Future-known Rejection')

    ax.plot(n_enrolled, external_rej, 'D-', color='#4A90D9', linewidth=2,
            markersize=5, label='External Unknown Rejection')

    # threshold 표시
    threshold = results_c.get('threshold', None)
    if threshold is not None:
        ax.axhline(y=threshold, color='gray', linestyle=':', linewidth=1,
                   alpha=0.5, label=f'Threshold={threshold:.3f}')

    ax.set_xlabel('Number of Enrolled Identities')
    ax.set_ylabel('Rate')
    ax.set_title('Protocol C: Sequential Enrollment Evaluation')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 2] saved: {save_path}")


# ============================================================
# Figure 3: Score Distribution (max cosine score)
# ============================================================

def plot_score_distributions(results_c: Dict, save_path: str):
    """
    step 0 vs final step의 max cosine score 분포.
    각 subplot에 known / future-known / external unknown 3개 히스토그램.
    Score = 각 probe의 gallery prototype 대비 max cosine similarity.
    """
    _setup_style()

    score_dists = results_c.get('score_distributions', {})
    if not score_dists:
        print("[FIGURE 3] no score distributions available, skipping")
        return

    step_keys = sorted(score_dists.keys(), key=lambda x: int(x.split('_')[1]))
    step_0_key = step_keys[0]
    step_final_key = step_keys[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    threshold = results_c.get('threshold', None)

    for ax, step_key, title_suffix in zip(
        axes, [step_0_key, step_final_key], ['Step 0 (Base only)', f'Final Step']
    ):
        dist = score_dists[step_key]
        bins = np.linspace(-0.2, 1.0, 60)

        if dist['known']:
            ax.hist(dist['known'], bins=bins, alpha=0.6, color='#2E8B57',
                    label=f'Known (n={len(dist["known"])})', density=True)
        if dist['future']:
            ax.hist(dist['future'], bins=bins, alpha=0.6, color='#DC3545',
                    label=f'Future-known (n={len(dist["future"])})', density=True)
        if dist['external']:
            ax.hist(dist['external'], bins=bins, alpha=0.6, color='#4A90D9',
                    label=f'External Unknown (n={len(dist["external"])})', density=True)

        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5,
                       label=f'Threshold={threshold:.3f}')

        ax.set_xlabel('Max Cosine Similarity Score')
        ax.set_title(title_suffix)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Density')
    fig.suptitle('Score Distribution: Max Cosine Similarity to Gallery', fontsize=13, y=1.02)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[FIGURE 3] saved: {save_path}")
