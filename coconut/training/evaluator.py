"""
Continual Learning Evaluator with Per-User Tracking
=====================================================
Properly tracks individual user performance over time for forgetting measurement
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import json
from datetime import datetime

from ..evaluation import ForgettingTracker, calculate_all_metrics, save_evaluation_curves
from ..openset import extract_features
from ..data import BaseVeinDataset, get_scr_transforms
from torch.utils.data import DataLoader, Subset


class ContinualLearningEvaluator:
    """
    Evaluator that tracks per-user performance over time.
    Calculates proper forgetting metrics based on individual user performance.
    """

    def __init__(self, config, test_file: str):
        """
        Initialize the evaluator with test data.

        Args:
            config: Configuration object
            test_file: Path to test set file
        """
        self.config = config
        self.test_file = test_file

        # Initialize forgetting tracker
        self.forgetting_tracker = ForgettingTracker(primary_metric='1-eer')

        # User-specific test data storage
        self.user_test_data = {}  # {user_id: {'paths': [...], 'indices': [...]}}
        self.registered_users = set()

        # Test dataset
        self.test_dataset = self._load_test_dataset()

        # Test step counter
        self.test_step = 0

        # History for analysis
        self.evaluation_history = []

    def _load_test_dataset(self) -> BaseVeinDataset:
        """Load the test dataset."""
        transform = get_scr_transforms(
            train=False,
            imside=self.config.dataset.height,
            channels=self.config.dataset.channels
        )

        # Convert Path object to string if needed
        test_file_path = str(self.test_file) if self.test_file else None

        dataset = BaseVeinDataset(
            paths=test_file_path,
            transform=transform,
            train=False,
            channels=self.config.dataset.channels,
            dual_views=False
        )

        print(f"[Evaluator] Loaded test dataset with {len(dataset)} samples")
        return dataset

    def prepare_user_test_data(self, user_id: int):
        """
        Prepare test data for a specific user.
        Filters test samples belonging to this user.

        Args:
            user_id: User identifier
        """
        if user_id in self.user_test_data:
            return  # Already prepared

        # Find all test samples for this user
        user_indices = []
        user_paths = []

        for idx in range(len(self.test_dataset)):
            label = int(self.test_dataset.labels[idx])
            if label == user_id:
                user_indices.append(idx)
                user_paths.append(self.test_dataset.paths[idx])

        self.user_test_data[user_id] = {
            'indices': user_indices,
            'paths': user_paths,
            'num_samples': len(user_indices)
        }

        print(f"[Evaluator] Prepared {len(user_indices)} test samples for user {user_id}")

    def register_user(self, user_id: int):
        """Register a new user for tracking."""
        self.registered_users.add(user_id)
        self.prepare_user_test_data(user_id)

    def evaluate_single_user(self, trainer, user_id: int,
                            use_tta: bool = False) -> Dict[str, float]:
        """
        Evaluate a single user's performance.

        Args:
            trainer: COCONUT trainer instance
            user_id: User to evaluate
            use_tta: Whether to use test-time augmentation

        Returns:
            Dictionary with biometric metrics
        """
        if user_id not in self.user_test_data:
            print(f"[Warning] No test data for user {user_id}")
            return {'eer': 1.0, 'tar_001': 0.0, 'margin': 0.0}

        # Get test samples for this user
        user_data = self.user_test_data[user_id]
        if not user_data['indices']:
            return {'eer': 1.0, 'tar_001': 0.0, 'margin': 0.0}

        # Create subset for this user
        user_subset = Subset(self.test_dataset, user_data['indices'])

        # Extract genuine scores (user authenticating as themselves)
        genuine_scores = []
        impostor_scores = []

        # Create dataloader
        loader = DataLoader(
            user_subset,
            batch_size=min(32, len(user_subset)),
            shuffle=False,
            num_workers=0
        )

        trainer.model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                # Move to device
                batch_images = batch_images.to(trainer.device)

                # Extract features using getFeatureCode for NCM compatibility (6144D)
                if use_tta and hasattr(trainer, 'use_tta') and trainer.use_tta:
                    # TTA: Multiple views with getFeatureCode
                    features = trainer._extract_features_with_tta(batch_images)
                else:
                    # Normal: Single view using getFeatureCode (6144D)
                    features = trainer.model.getFeatureCode(batch_images)

                # Calculate similarities
                for feat in features:
                    feat = feat.unsqueeze(0)  # Add batch dimension

                    # Genuine: similarity to correct user
                    sim_genuine = trainer.ncm.similarity(feat, user_id)
                    if sim_genuine is not None:
                        genuine_scores.append(float(sim_genuine))

                    # Impostor: similarities to other users
                    for other_id in self.registered_users:
                        if other_id != user_id:
                            sim_impostor = trainer.ncm.similarity(feat, other_id)
                            if sim_impostor is not None:
                                impostor_scores.append(float(sim_impostor))

        # Calculate metrics
        if not genuine_scores or not impostor_scores:
            print(f"[Warning] Insufficient scores for user {user_id}")
            return {
                'eer': 1.0, 'tar_001': 0.0, 'margin': 0.0,
                'genuine_scores': [], 'impostor_scores': []
            }

        genuine_np = np.array(genuine_scores)
        impostor_np = np.array(impostor_scores)

        metrics = calculate_all_metrics(genuine_np, impostor_np)

        # Add raw scores to metrics for later aggregation
        metrics['genuine_scores'] = genuine_scores
        metrics['impostor_scores'] = impostor_scores

        return metrics

    def evaluate_all_users(self, trainer, experience_id: int,
                          use_tta: bool = False,
                          save_curves: bool = False,
                          curves_dir: Optional[str] = None) -> Dict:
        """
        Evaluate all registered users individually.

        Args:
            trainer: COCONUT trainer instance
            experience_id: Current experience count
            use_tta: Whether to use test-time augmentation
            save_curves: Whether to save ROC/DET curves
            curves_dir: Directory to save curves (if save_curves=True)

        Returns:
            Dictionary with evaluation results
        """
        self.test_step += 1
        print(f"\n{'='*60}")
        print(f" Evaluating All {len(self.registered_users)} Users (Step {self.test_step})")
        print(f"{'='*60}")

        # Collect all scores for aggregate curves
        all_genuine_scores = []
        all_impostor_scores = []

        # Evaluate each user
        for user_id in sorted(self.registered_users):
            print(f"\nEvaluating User {user_id}...", end=' ')

            # Get performance metrics
            metrics = self.evaluate_single_user(trainer, user_id, use_tta)

            # Collect scores for aggregate analysis
            if 'genuine_scores' in metrics:
                all_genuine_scores.extend(metrics['genuine_scores'])
                all_impostor_scores.extend(metrics['impostor_scores'])

            # Remove raw scores before updating tracker (to save memory)
            metrics_for_tracker = {k: v for k, v in metrics.items()
                                  if k not in ['genuine_scores', 'impostor_scores']}

            # Update forgetting tracker
            self.forgetting_tracker.update_user_performance(
                user_id=user_id,
                performance_dict=metrics_for_tracker,
                test_step=self.test_step,
                experience_id=experience_id
            )

            print(f"1-EER: {metrics.get('1-eer', 0):.3f}, "
                  f"TAR@1%: {metrics.get('tar_001', 0):.3f}")

        # Generate report
        report = self.forgetting_tracker.get_report(self.test_step, experience_id)

        # Save evaluation curves if requested
        if save_curves and all_genuine_scores and all_impostor_scores:
            if curves_dir is None:
                curves_dir = f"evaluation_curves/exp_{experience_id}"

            print(f"\n📊 Generating evaluation curves...")

            # Calculate overall EER and threshold
            genuine_np = np.array(all_genuine_scores)
            impostor_np = np.array(all_impostor_scores)
            overall_metrics = calculate_all_metrics(genuine_np, impostor_np)

            # Save curves and tables
            saved_files = save_evaluation_curves(
                genuine_np, impostor_np,
                output_dir=curves_dir,
                experience_id=experience_id,
                eer_value=overall_metrics['eer'],
                threshold=None,  # Will be calculated if needed
                n_points=1000
            )

            # Add saved file paths to report
            report['evaluation_curves'] = saved_files

            # Also save aggregated metrics
            report['aggregate_metrics'] = overall_metrics

        # Print summary
        self.forgetting_tracker.print_summary(self.test_step)

        # Store in history
        self.evaluation_history.append(report)

        return report

    def get_forgetting_measure(self, metric: str = '1-eer') -> float:
        """Get average forgetting across all users."""
        return self.forgetting_tracker.get_average_forgetting(metric)

    def get_average_performance(self, metric: str = '1-eer') -> float:
        """Get average current performance across all users."""
        performances = []
        for user_id, user_data in self.forgetting_tracker.performance_matrix.items():
            if user_data['current_performance']:
                perf = user_data['current_performance'].get(metric)
                if perf is not None:
                    performances.append(perf)

        return np.mean(performances) if performances else 0.0

    def get_user_trajectory(self, user_id: int, metric: str = '1-eer') -> List[float]:
        """Get performance trajectory for a specific user."""
        return self.forgetting_tracker.get_performance_trajectory(user_id, metric)

    def save_results(self, save_dir: str):
        """Save evaluation results to files."""
        os.makedirs(save_dir, exist_ok=True)

        # Save forgetting tracker state
        tracker_path = os.path.join(save_dir, f'forgetting_tracker_step_{self.test_step:03d}.json')
        self.forgetting_tracker.save_checkpoint(tracker_path)

        # Save evaluation history
        history_path = os.path.join(save_dir, f'evaluation_history_step_{self.test_step:03d}.json')
        with open(history_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, default=str)

        print(f"[Evaluator] Results saved to {save_dir}")

    def create_performance_matrix_csv(self, save_path: str):
        """Export performance matrix to CSV for analysis."""
        import csv

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            max_steps = max(len(data['evaluations'])
                          for data in self.forgetting_tracker.performance_matrix.values())
            header = ['User'] + [f'Step_{i+1}' for i in range(max_steps)]
            writer.writerow(header)

            # Data rows
            for user_id in sorted(self.forgetting_tracker.performance_matrix.keys()):
                user_data = self.forgetting_tracker.performance_matrix[user_id]
                row = [user_id]

                for eval_record in user_data['evaluations']:
                    metric_value = eval_record['metrics'].get('1-eer', 0)
                    row.append(f"{metric_value:.4f}")

                # Pad with empty cells if needed
                while len(row) < len(header):
                    row.append('')

                writer.writerow(row)

        print(f"[Evaluator] Performance matrix exported to {save_path}")

    def plot_forgetting_curves(self, save_path: str, top_k: int = 10):
        """Plot forgetting curves for top-k users with highest forgetting."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            # Get users with highest forgetting
            forgetting_dict = self.forgetting_tracker.calculate_forgetting()
            sorted_users = sorted(forgetting_dict.items(), key=lambda x: x[1], reverse=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Performance trajectories
            for user_id, forget_val in sorted_users[:top_k]:
                trajectory = self.get_user_trajectory(user_id)
                if trajectory:
                    steps = list(range(1, len(trajectory) + 1))
                    ax1.plot(steps, trajectory, marker='o', label=f'User {user_id}')

            ax1.set_xlabel('Test Step')
            ax1.set_ylabel('1-EER Performance')
            ax1.set_title(f'Performance Trajectories (Top {top_k} Forgetting)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Forgetting distribution
            forgetting_values = list(forgetting_dict.values())
            ax2.hist(forgetting_values, bins=20, edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(forgetting_values), color='red',
                       linestyle='--', label=f'Mean: {np.mean(forgetting_values):.3f}')
            ax2.set_xlabel('Forgetting Value')
            ax2.set_ylabel('Number of Users')
            ax2.set_title('Forgetting Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Evaluator] Forgetting curves saved to {save_path}")

        except ImportError:
            print("[Warning] matplotlib not available, skipping plot")
        except Exception as e:
            print(f"[Warning] Error creating plot: {e}")