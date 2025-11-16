"""
Forgetting Tracker for Continual Learning
==========================================
Tracks per-user performance over time and calculates forgetting metrics
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os
from datetime import datetime


class ForgettingTracker:
    """
    Tracks performance metrics for each user over time.
    Calculates forgetting as: F_i = A_i^max - A_i^current
    """

    def __init__(self, primary_metric: str = '1-eer'):
        """
        Initialize the forgetting tracker.

        Args:
            primary_metric: Primary metric to use for forgetting calculation
                          Options: '1-eer', 'tar_001', 'tar_0001', 'margin'
        """
        self.primary_metric = primary_metric

        # Main data structure: {user_id: {'evaluations': [...], 'max_performance': {...}}}
        self.performance_matrix = defaultdict(lambda: {
            'evaluations': [],
            'max_performance': {},
            'current_performance': {},
            'first_seen_step': None
        })

        # Test history for tracking when evaluations occurred
        self.test_history = []

        # Metadata
        self.total_users = 0
        self.total_evaluations = 0

    def update_user_performance(self, user_id: int, performance_dict: Dict,
                               test_step: int, experience_id: int):
        """
        Update performance for a specific user.

        Args:
            user_id: User identifier
            performance_dict: Dictionary containing metrics
                            {'eer': 0.02, 'tar_001': 0.98, 'tar_0001': 0.95, 'margin': 0.5}
            test_step: Current test step (e.g., 1, 2, 3...)
            experience_id: Current experience/user count
        """
        user_data = self.performance_matrix[user_id]

        # Record first appearance
        if user_data['first_seen_step'] is None:
            user_data['first_seen_step'] = test_step

        # Create evaluation record
        eval_record = {
            'step': test_step,
            'experience': experience_id,
            'metrics': performance_dict.copy(),
            'timestamp': datetime.now().isoformat()
        }

        # Note: '1-eer' is already provided by calculate_all_metrics, no need to recalculate

        user_data['evaluations'].append(eval_record)

        # Update current performance
        user_data['current_performance'] = eval_record['metrics']

        # Update max performance
        if not user_data['max_performance']:
            user_data['max_performance'] = eval_record['metrics'].copy()
        else:
            for metric, value in eval_record['metrics'].items():
                # For all metrics except EER, higher is better
                if metric == 'eer':
                    if value < user_data['max_performance'].get(metric, float('inf')):
                        user_data['max_performance'][metric] = value
                else:
                    if value > user_data['max_performance'].get(metric, -float('inf')):
                        user_data['max_performance'][metric] = value

        self.total_evaluations += 1

    def calculate_forgetting(self, metric: Optional[str] = None) -> Dict[int, float]:
        """
        Calculate forgetting for each user.

        Args:
            metric: Metric to use for forgetting calculation (default: primary_metric)

        Returns:
            Dictionary mapping user_id to forgetting value
        """
        metric = metric or self.primary_metric
        forgetting = {}

        for user_id, user_data in self.performance_matrix.items():
            if not user_data['evaluations']:
                continue

            max_val = user_data['max_performance'].get(metric)
            curr_val = user_data['current_performance'].get(metric)

            if max_val is not None and curr_val is not None:
                # For EER, forgetting is current - max (since lower is better)
                if metric == 'eer':
                    forgetting[user_id] = curr_val - max_val
                # For other metrics, forgetting is max - current (since higher is better)
                else:
                    forgetting[user_id] = max_val - curr_val
            else:
                forgetting[user_id] = 0.0

        return forgetting

    def get_average_forgetting(self, metric: Optional[str] = None) -> float:
        """Calculate average forgetting across all users."""
        forgetting = self.calculate_forgetting(metric)
        if not forgetting:
            return 0.0
        return np.mean(list(forgetting.values()))

    def get_group_forgetting(self, group_size: int = 10,
                           metric: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate forgetting by groups of users.

        Args:
            group_size: Number of users per group
            metric: Metric to use for calculation

        Returns:
            Dictionary with group names and average forgetting
        """
        forgetting = self.calculate_forgetting(metric)
        groups = defaultdict(list)

        for user_id, forget_val in forgetting.items():
            group_idx = (user_id - 1) // group_size
            group_name = f"users_{group_idx * group_size + 1}-{(group_idx + 1) * group_size}"
            groups[group_name].append(forget_val)

        group_forgetting = {}
        for group_name, values in groups.items():
            group_forgetting[group_name] = np.mean(values) if values else 0.0

        return group_forgetting

    def get_performance_trajectory(self, user_id: int, metric: Optional[str] = None) -> List[float]:
        """Get performance trajectory for a specific user."""
        metric = metric or self.primary_metric
        user_data = self.performance_matrix.get(user_id)

        if not user_data or not user_data['evaluations']:
            return []

        trajectory = []
        for eval_record in user_data['evaluations']:
            value = eval_record['metrics'].get(metric)
            if value is not None:
                trajectory.append(value)

        return trajectory

    def get_report(self, test_step: int, experience_id: int) -> Dict:
        """
        Generate comprehensive forgetting report.

        Args:
            test_step: Current test step
            experience_id: Current experience count

        Returns:
            Dictionary with complete analysis
        """
        forgetting = self.calculate_forgetting()

        # Per-user analysis
        user_analysis = {}
        for user_id, forget_val in forgetting.items():
            user_data = self.performance_matrix[user_id]
            user_analysis[user_id] = {
                'forgetting': forget_val,
                'max_performance': user_data['max_performance'].get(self.primary_metric, 0),
                'current_performance': user_data['current_performance'].get(self.primary_metric, 0),
                'num_evaluations': len(user_data['evaluations']),
                'first_seen': user_data['first_seen_step']
            }

        # Group analysis
        group_analysis = self.get_group_forgetting()

        # Overall statistics
        forgetting_values = list(forgetting.values())

        report = {
            'test_step': test_step,
            'total_users': experience_id,
            'evaluated_users': len(forgetting),
            'primary_metric': self.primary_metric,
            'per_user': user_analysis,
            'group_analysis': group_analysis,
            'overall': {
                'mean_forgetting': np.mean(forgetting_values) if forgetting_values else 0.0,
                'std_forgetting': np.std(forgetting_values) if forgetting_values else 0.0,
                'max_forgetting': np.max(forgetting_values) if forgetting_values else 0.0,
                'min_forgetting': np.min(forgetting_values) if forgetting_values else 0.0
            },
            'timestamp': datetime.now().isoformat()
        }

        return report

    def save_checkpoint(self, filepath: str):
        """Save current state to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        data = {
            'primary_metric': self.primary_metric,
            'performance_matrix': dict(self.performance_matrix),
            'test_history': self.test_history,
            'total_users': self.total_users,
            'total_evaluations': self.total_evaluations
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_checkpoint(self, filepath: str):
        """Load state from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.primary_metric = data['primary_metric']
        self.performance_matrix = defaultdict(lambda: {
            'evaluations': [],
            'max_performance': {},
            'current_performance': {},
            'first_seen_step': None
        }, data['performance_matrix'])
        self.test_history = data['test_history']
        self.total_users = data['total_users']
        self.total_evaluations = data['total_evaluations']

    def print_summary(self, test_step: int, top_k: int = 10):
        """Print formatted summary of forgetting analysis."""
        report = self.get_report(test_step, len(self.performance_matrix))

        print("\n" + "="*60)
        print(f" Forgetting Analysis Report (Step {test_step})")
        print("="*60)

        print(f"\n📊 Overall Statistics:")
        print(f"  Evaluated Users: {report['evaluated_users']}")
        print(f"  Mean Forgetting: {report['overall']['mean_forgetting']:.4f}")
        print(f"  Std Forgetting: {report['overall']['std_forgetting']:.4f}")
        print(f"  Max Forgetting: {report['overall']['max_forgetting']:.4f}")

        if report['group_analysis']:
            print(f"\n📈 Group Analysis:")
            for group_name, forget_val in sorted(report['group_analysis'].items()):
                print(f"  {group_name}: {forget_val:.4f}")

        # Show top users with highest forgetting
        if report['per_user']:
            sorted_users = sorted(report['per_user'].items(),
                                key=lambda x: x[1]['forgetting'],
                                reverse=True)

            print(f"\n⚠️ Top {min(top_k, len(sorted_users))} Users with Highest Forgetting:")
            print("  User | Max Perf | Curr Perf | Forgetting")
            print("  " + "-"*45)

            for user_id, data in sorted_users[:top_k]:
                print(f"  {user_id:4d} | {data['max_performance']:8.4f} | "
                      f"{data['current_performance']:9.4f} | {data['forgetting']:10.4f}")

        print("="*60)