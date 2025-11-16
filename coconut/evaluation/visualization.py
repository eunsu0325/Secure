"""
Biometric evaluation visualization tools for ROC/DET curves and FAR tables.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import json
from scipy import interpolate


def plot_roc_curve(
    far_array: np.ndarray,
    tar_array: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show_eer: bool = True,
    eer_value: Optional[float] = None
) -> plt.Figure:
    """
    Plot ROC curve (FAR vs TAR).

    Args:
        far_array: False Accept Rate values
        tar_array: True Accept Rate values
        title: Plot title
        save_path: Path to save the figure
        show_eer: Whether to show EER point
        eer_value: EER value to display

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve
    ax.plot(far_array, tar_array, 'b-', linewidth=2, label='ROC Curve')

    # Add EER point if provided
    if show_eer and eer_value is not None:
        # Find point closest to EER
        eer_idx = np.argmin(np.abs(far_array - eer_value))
        ax.plot(far_array[eer_idx], tar_array[eer_idx], 'ro', markersize=10,
                label=f'EER = {eer_value:.3%}')

        # Add diagonal line for EER
        ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, linewidth=1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Labels and title
    ax.set_xlabel('False Accept Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Accept Rate (TAR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add legend
    ax.legend(loc='lower right', fontsize=10)

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    plt.close()
    return fig


def plot_det_curve(
    far_array: np.ndarray,
    frr_array: np.ndarray,
    title: str = "DET Curve",
    save_path: Optional[str] = None,
    show_eer: bool = True,
    eer_value: Optional[float] = None
) -> plt.Figure:
    """
    Plot DET curve (FAR vs FRR in log scale).

    Args:
        far_array: False Accept Rate values
        frr_array: False Reject Rate values (1 - TAR)
        title: Plot title
        save_path: Path to save the figure
        show_eer: Whether to show EER point
        eer_value: EER value to display

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to percentage for better readability
    far_percent = far_array * 100
    frr_percent = frr_array * 100

    # Filter out zeros for log scale
    valid_mask = (far_percent > 0) & (frr_percent > 0)
    far_plot = far_percent[valid_mask]
    frr_plot = frr_percent[valid_mask]

    # Plot DET curve
    ax.loglog(far_plot, frr_plot, 'b-', linewidth=2, label='DET Curve')

    # Add EER point if provided
    if show_eer and eer_value is not None:
        eer_percent = eer_value * 100
        if eer_percent > 0:
            ax.loglog(eer_percent, eer_percent, 'ro', markersize=10,
                     label=f'EER = {eer_value:.3%}')

            # Add diagonal line for EER
            min_val = min(far_plot.min(), frr_plot.min())
            max_val = max(far_plot.max(), frr_plot.max())
            ax.loglog([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)

    # Add grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    # Labels and title
    ax.set_xlabel('False Accept Rate (FAR) [%]', fontsize=12)
    ax.set_ylabel('False Reject Rate (FRR) [%]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set axis limits (0.01% to 100%)
    ax.set_xlim([0.01, 100])
    ax.set_ylim([0.01, 100])

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DET curve saved to: {save_path}")

    plt.close()
    return fig


def generate_far_tar_table(
    far_array: np.ndarray,
    tar_array: np.ndarray,
    thresholds: np.ndarray,
    far_points: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate TAR values at specific FAR points.

    Args:
        far_array: False Accept Rate values
        tar_array: True Accept Rate values
        thresholds: Threshold values corresponding to FAR/TAR
        far_points: List of FAR points to evaluate (default: common points)
        save_path: Path to save the table as CSV

    Returns:
        DataFrame with FAR points and corresponding TAR values
    """
    if far_points is None:
        # Common FAR points used in biometric papers
        far_points = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

    results = []

    # Interpolate to get TAR at specific FAR points
    # Sort by FAR for interpolation
    sorted_idx = np.argsort(far_array)
    far_sorted = far_array[sorted_idx]
    tar_sorted = tar_array[sorted_idx]
    thresh_sorted = thresholds[sorted_idx]

    # Remove duplicates for interpolation
    unique_mask = np.append(True, far_sorted[1:] != far_sorted[:-1])
    far_unique = far_sorted[unique_mask]
    tar_unique = tar_sorted[unique_mask]
    thresh_unique = thresh_sorted[unique_mask]

    # Create interpolation functions
    if len(far_unique) > 1:
        tar_interp = interpolate.interp1d(
            far_unique, tar_unique,
            kind='linear', bounds_error=False, fill_value=(tar_unique[0], tar_unique[-1])
        )
        thresh_interp = interpolate.interp1d(
            far_unique, thresh_unique,
            kind='linear', bounds_error=False, fill_value=(thresh_unique[0], thresh_unique[-1])
        )
    else:
        tar_interp = lambda x: tar_unique[0]
        thresh_interp = lambda x: thresh_unique[0]

    for far_point in far_points:
        tar_value = float(tar_interp(far_point))
        threshold = float(thresh_interp(far_point))

        results.append({
            'FAR (%)': f"{far_point * 100:.4f}%",
            'FAR': far_point,
            'TAR (%)': f"{tar_value * 100:.2f}%",
            'TAR': tar_value,
            'FRR (%)': f"{(1 - tar_value) * 100:.2f}%",
            'FRR': 1 - tar_value,
            'Threshold': f"{threshold:.4f}"
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save if path provided
    if save_path:
        # Save as CSV
        csv_path = Path(save_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"FAR-TAR table saved to: {csv_path}")

        # Also save as formatted text
        txt_path = Path(save_path).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FAR vs TAR Performance Table\n")
            f.write("="*80 + "\n\n")
            f.write(df[['FAR (%)', 'TAR (%)', 'FRR (%)', 'Threshold']].to_string(index=False))
        print(f"FAR-TAR table (text) saved to: {txt_path}")

    return df


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Score Distributions",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot genuine and impostor score distributions.

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts
        threshold: Decision threshold to display
        title: Plot title
        save_path: Path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histograms
    bins = np.linspace(
        min(genuine_scores.min(), impostor_scores.min()),
        max(genuine_scores.max(), impostor_scores.max()),
        50
    )

    ax.hist(impostor_scores, bins=bins, alpha=0.5, label='Impostor',
            color='red', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(genuine_scores, bins=bins, alpha=0.5, label='Genuine',
            color='green', density=True, edgecolor='black', linewidth=0.5)

    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold:.3f}')

    # Labels and title
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text = (
        f"Genuine: μ={genuine_scores.mean():.3f}, σ={genuine_scores.std():.3f}\n"
        f"Impostor: μ={impostor_scores.mean():.3f}, σ={impostor_scores.std():.3f}\n"
        f"Margin: {genuine_scores.mean() - impostor_scores.mean():.3f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")

    plt.close()
    return fig


def save_evaluation_curves(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    output_dir: str,
    experience_id: Optional[int] = None,
    eer_value: Optional[float] = None,
    threshold: Optional[float] = None,
    n_points: int = 1000
) -> Dict:
    """
    Generate and save all evaluation curves and tables.

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts
        output_dir: Directory to save outputs
        experience_id: Current experience/task ID
        eer_value: Pre-computed EER value
        threshold: Decision threshold
        n_points: Number of points for curves

    Returns:
        Dictionary with paths to saved files
    """
    from ..evaluation import compute_roc_curve, calculate_eer

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add experience suffix if provided
    suffix = f"_exp{experience_id}" if experience_id is not None else ""

    # Compute curves
    far_array, tar_array, thresholds = compute_roc_curve(
        genuine_scores, impostor_scores, n_points
    )
    frr_array = 1 - tar_array

    # Calculate EER if not provided
    if eer_value is None:
        eer_value = calculate_eer(genuine_scores, impostor_scores)

    saved_files = {}

    # 1. Save ROC curve
    roc_path = output_path / f"roc_curve{suffix}.png"
    plot_roc_curve(
        far_array, tar_array,
        title=f"ROC Curve{' - Experience ' + str(experience_id) if experience_id else ''}",
        save_path=str(roc_path),
        show_eer=True,
        eer_value=eer_value
    )
    saved_files['roc_curve'] = str(roc_path)

    # 2. Save DET curve
    det_path = output_path / f"det_curve{suffix}.png"
    plot_det_curve(
        far_array, frr_array,
        title=f"DET Curve{' - Experience ' + str(experience_id) if experience_id else ''}",
        save_path=str(det_path),
        show_eer=True,
        eer_value=eer_value
    )
    saved_files['det_curve'] = str(det_path)

    # 3. Save FAR-TAR table
    table_path = output_path / f"far_tar_table{suffix}"
    far_points = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
    df_table = generate_far_tar_table(
        far_array, tar_array, thresholds,
        far_points=far_points,
        save_path=str(table_path)
    )
    saved_files['far_tar_table_csv'] = str(table_path.with_suffix('.csv'))
    saved_files['far_tar_table_txt'] = str(table_path.with_suffix('.txt'))

    # 4. Save score distributions
    dist_path = output_path / f"score_distributions{suffix}.png"
    plot_score_distributions(
        genuine_scores, impostor_scores,
        threshold=threshold,
        title=f"Score Distributions{' - Experience ' + str(experience_id) if experience_id else ''}",
        save_path=str(dist_path)
    )
    saved_files['score_distributions'] = str(dist_path)

    # 5. Save raw data as JSON for reproducibility
    data_path = output_path / f"evaluation_data{suffix}.json"
    evaluation_data = {
        'eer': float(eer_value),
        'threshold': float(threshold) if threshold else None,
        'genuine_scores_stats': {
            'mean': float(genuine_scores.mean()),
            'std': float(genuine_scores.std()),
            'min': float(genuine_scores.min()),
            'max': float(genuine_scores.max()),
            'count': len(genuine_scores)
        },
        'impostor_scores_stats': {
            'mean': float(impostor_scores.mean()),
            'std': float(impostor_scores.std()),
            'min': float(impostor_scores.min()),
            'max': float(impostor_scores.max()),
            'count': len(impostor_scores)
        },
        'far_tar_table': df_table[['FAR', 'TAR', 'FRR']].to_dict('records')
    }

    with open(data_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    saved_files['evaluation_data'] = str(data_path)

    print(f"\n📊 Evaluation curves and tables saved to: {output_path}")
    print(f"   - ROC curve: {roc_path.name}")
    print(f"   - DET curve: {det_path.name}")
    print(f"   - FAR-TAR table: {table_path.name}.csv/.txt")
    print(f"   - Score distributions: {dist_path.name}")
    print(f"   - Raw data: {data_path.name}")

    return saved_files