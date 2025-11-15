#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCONUT Training Script
Compact Online Continual Neural User Templates
Main training script for palm vein recognition with continual learning
"""

import os
import argparse
import time
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

# Project imports
from config import ConfigParser
from utils.pretrained_loader import PretrainedLoader
from models import ccnet, MyDataset, get_scr_transforms
from coconut import (
    ExperienceStream,
    ClassBalancedBuffer,
    NCMClassifier,
    COCONUTTrainer
)
from utils.utils_openset import predict_batch, predict_batch_tta


def compute_safe_base_id(*txt_files):
    """Compute safe BASE_ID from maximum user ID in all txt files"""
    max_id = 0
    for path in txt_files:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    lab = int(parts[1])
                except ValueError:
                    continue
                max_id = max(max_id, lab)
    base = max_id + 1
    print(f"[BASE_ID] max_user_id={max_id}, BASE_ID={base}")
    return base


def purge_negatives(memory_buffer, base_id):
    """Remove all negative classes from memory buffer"""
    to_del = [int(c) for c in list(memory_buffer.seen_classes) if int(c) >= base_id]
    removed = 0
    for c in to_del:
        if c in memory_buffer.buffer_groups:
            removed += len(memory_buffer.buffer_groups[c].buffer)
            del memory_buffer.buffer_groups[c]
        memory_buffer.seen_classes.discard(c)
    print(f"Purged {len(to_del)} negative classes ({removed} samples)")


class ContinualLearningEvaluator:
    """Evaluator for continual learning metrics"""

    def __init__(self, num_experiences: int):
        self.num_experiences = num_experiences
        self.accuracy_history = defaultdict(list)
        self.openset_history = []

    def update(self, experience_id: int, accuracy: float):
        """Update accuracy after training experience"""
        self.accuracy_history[experience_id].append(accuracy)

    def update_openset(self, metrics: dict):
        """Update open-set metrics"""
        self.openset_history.append(metrics)

    def get_average_accuracy(self) -> float:
        """Calculate average accuracy across all experiences"""
        all_accs = []
        for accs in self.accuracy_history.values():
            if accs:
                all_accs.append(accs[-1])
        return np.mean(all_accs) if all_accs else 0.0

    def get_forgetting_measure(self) -> float:
        """Calculate forgetting measure"""
        forgetting = []
        for exp_id, accs in self.accuracy_history.items():
            if len(accs) > 1:
                max_acc = max(accs[:-1])
                curr_acc = accs[-1]
                forgetting.append(max_acc - curr_acc)
        return np.mean(forgetting) if forgetting else 0.0


def evaluate_on_test_set(trainer: COCONUTTrainer, config, openset_mode=False) -> tuple:
    """
    Evaluate on test set with optional open-set evaluation.

    Returns:
        (accuracy, openset_metrics) if openset_mode else (accuracy, None)
    """
    test_dataset = MyDataset(
        txt=config.dataset.test_set_file,
        transforms=get_scr_transforms(
            train=False,
            imside=config.dataset.height,
            channels=config.dataset.channels
        ),
        train=False,
        imside=config.dataset.height,
        outchannels=config.dataset.channels
    )

    known_classes = set(trainer.ncm.class_means_dict.keys())
    if not known_classes:
        return (0.0, {}) if openset_mode else (0.0, None)

    # Check TTA settings
    use_tta = (hasattr(trainer, 'use_tta') and trainer.use_tta and
               hasattr(config, 'openset') and config.openset.tta_n_views > 1)

    if openset_mode and trainer.openset_enabled:
        # Open-set evaluation
        known_indices = []
        unknown_indices = []

        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                known_indices.append(i)
            else:
                unknown_indices.append(i)

        # Evaluate on known classes
        if known_indices:
            known_subset = Subset(test_dataset, known_indices)
            accuracy = trainer.evaluate(known_subset)
        else:
            accuracy = 0.0

        openset_metrics = {}

        # Calculate open-set metrics
        if known_indices and trainer.ncm.tau_s is not None:
            if len(known_indices) > 500:
                known_indices = np.random.choice(known_indices, 500, replace=False)

            known_paths = [test_dataset.images_path[i] for i in known_indices]
            known_labels = [int(test_dataset.images_label[i]) for i in known_indices]

            # Predict with or without TTA
            if use_tta:
                preds = predict_batch_tta(
                    trainer.model, trainer.ncm,
                    known_paths, trainer.test_transform, trainer.device,
                    n_views=config.openset.tta_n_views,
                    include_original=config.openset.tta_include_original,
                    agree_k=config.openset.tta_agree_k,
                    aug_strength=config.openset.tta_augmentation_strength,
                    img_size=config.dataset.height,
                    channels=config.dataset.channels
                )
            else:
                preds = predict_batch(
                    trainer.model, trainer.ncm,
                    known_paths, trainer.test_transform, trainer.device,
                    channels=config.dataset.channels
                )

            correct = sum(1 for p, l in zip(preds, known_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)

            openset_metrics['TAR'] = correct / max(1, len(preds))
            openset_metrics['FRR'] = rejected / max(1, len(preds))

        # Evaluate on unknown classes
        if unknown_indices and trainer.ncm.tau_s is not None:
            if len(unknown_indices) > 500:
                unknown_indices = np.random.choice(unknown_indices, 500, replace=False)

            unknown_paths = [test_dataset.images_path[i] for i in unknown_indices]

            if use_tta:
                preds = predict_batch_tta(
                    trainer.model, trainer.ncm,
                    unknown_paths, trainer.test_transform, trainer.device,
                    n_views=config.openset.tta_n_views,
                    include_original=config.openset.tta_include_original,
                    agree_k=config.openset.tta_agree_k,
                    aug_strength=config.openset.tta_augmentation_strength,
                    img_size=config.dataset.height,
                    channels=config.dataset.channels
                )
            else:
                preds = predict_batch(
                    trainer.model, trainer.ncm,
                    unknown_paths, trainer.test_transform, trainer.device,
                    channels=config.dataset.channels
                )

            openset_metrics['TRR_unknown'] = sum(1 for p in preds if p == -1) / len(preds)
            openset_metrics['FAR_unknown'] = 1 - openset_metrics['TRR_unknown']

        openset_metrics['tau_s'] = trainer.ncm.tau_s
        openset_metrics['tta_enabled'] = use_tta

        print(f"Evaluating on {len(known_indices)} known + {len(unknown_indices)} unknown samples"
              + (" (TTA enabled)" if use_tta else ""))

        return accuracy, openset_metrics

    else:
        # Closed-set evaluation
        filtered_indices = []
        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                filtered_indices.append(i)

        if not filtered_indices:
            return (0.0, None)

        filtered_test = Subset(test_dataset, filtered_indices)
        print(f"Evaluating on {len(filtered_indices)} test samples from {len(known_classes)} classes")

        accuracy = trainer.evaluate(filtered_test)
        return accuracy, None


def main(args):
    """Main training function"""

    print("\n" + "="*60)
    print(" COCONUT Training")
    print(" Compact Online Continual Neural User Templates")
    print("="*60 + "\n")

    # 1. Load configuration
    config = ConfigParser(args.config)
    config_obj = config.get_config()
    print(f"Using config: {args.config}")
    print(config)

    # Add seed to config if not present
    if not hasattr(config_obj.training, 'seed'):
        config_obj.training.seed = args.seed

    # Calculate BASE_ID
    config_obj.negative.base_id = compute_safe_base_id(
        config_obj.dataset.train_set_file,
        config_obj.dataset.test_set_file
    )

    # Check open-set mode
    openset_enabled = hasattr(config_obj, 'openset') and config_obj.openset.enabled
    if openset_enabled:
        print("\n========== OPEN-SET MODE ENABLED ==========")
        print(f"   Warmup users: {config_obj.openset.warmup_users}")
        print(f"   Initial tau: {config_obj.openset.initial_tau}")
        print(f"   Threshold mode: {config_obj.openset.threshold_mode.upper()}")
        print(f"   Target FAR: {config_obj.openset.target_far*100:.1f}%")
        print("==========================================\n")

    # Check ProxyAnchor settings
    if hasattr(config_obj.training, 'use_proxy_anchor') and config_obj.training.use_proxy_anchor:
        print("\n========== PROXY ANCHOR LOSS ENABLED ==========")
        print(f"   Margin (δ): {config_obj.training.proxy_margin}")
        print(f"   Alpha (α): {config_obj.training.proxy_alpha}")
        print(f"   Lambda: {config_obj.training.proxy_lambda}")
        print("===============================================\n")

    # Setup device
    device = torch.device(
        f"cuda:{config_obj.training.gpu_ids}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print(f'Device: {device}')

    # 2. Create results directory
    results_dir = os.path.join(config_obj.training.results_path, 'coconut_results')
    os.makedirs(results_dir, exist_ok=True)

    # 3. Initialize data stream
    print("\n=== Initializing Data Stream ===")
    data_stream = ExperienceStream(
        train_file=config_obj.dataset.train_set_file,
        negative_file=config_obj.dataset.negative_samples_file,
        num_negative_classes=config_obj.dataset.num_negative_classes,
        base_id=config_obj.negative.base_id
    )

    stats = data_stream.get_statistics()
    print(f"Total users: {stats['num_users']}")
    print(f"Samples per user: {stats['samples_per_user']}")
    print(f"Negative samples: {stats['negative_samples']}")

    # 4. Initialize model and components
    print("\n=== Initializing Model and Components ===")

    # CCNet model
    model = ccnet(
        weight=config_obj.model.competition_weight,
        use_projection=config_obj.model.use_projection,
        projection_dim=config_obj.model.projection_dim
    )

    # Load pretrained weights if available
    if config_obj.model.use_pretrained and config_obj.model.pretrained_path:
        if config_obj.model.pretrained_path.exists():
            print(f"\nLoading pretrained weights...")
            loader = PretrainedLoader()
            try:
                model = loader.load_ccnet_pretrained(
                    model=model,
                    checkpoint_path=config_obj.model.pretrained_path,
                    device=device,
                    verbose=True
                )
                print("Pretrained weights loaded successfully!")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                print("Continuing with random initialization...")
        else:
            print(f"Pretrained path not found: {config_obj.model.pretrained_path}")
    else:
        print("Starting from random initialization")

    model = model.to(device)

    # Initialize COCONUT components
    ncm_classifier = NCMClassifier(normalize=True).to(device)
    memory_buffer = ClassBalancedBuffer(
        max_size=config_obj.training.memory_size,
        min_samples_per_class=config_obj.training.min_samples_per_class
    )

    # Initialize trainer
    trainer = COCONUTTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config_obj,
        device=device
    )

    # 5. Initialize with negative samples
    print("\n=== Initializing with Negative Samples ===")
    neg_paths, neg_labels = data_stream.get_negative_samples()

    if len(neg_paths) > config_obj.training.memory_batch_size:
        selected_indices = np.random.choice(
            len(neg_paths),
            size=config_obj.training.memory_batch_size,
            replace=False
        )
        neg_paths = [neg_paths[i] for i in selected_indices]
        neg_labels = [neg_labels[i] for i in selected_indices]

    memory_buffer.update_from_dataset(neg_paths, neg_labels)
    print(f"Initial buffer size: {len(memory_buffer)}")
    print("NCM starts empty - no fake class contamination")

    # 6. Initialize evaluator
    evaluator = ContinualLearningEvaluator(num_experiences=config_obj.training.num_experiences)

    # 7. Training history
    training_history = {
        'losses': [],
        'accuracies': [],
        'forgetting_measures': [],
        'memory_sizes': [],
        'openset_metrics': [],
        'pretrained_used': config_obj.model.use_pretrained,
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed,
        'config_path': args.config
    }

    # 8. Start continual learning
    print("\n=== Starting Continual Learning ===")
    print(f"Total experiences: {config_obj.training.num_experiences}")
    print(f"Evaluation interval: every {config_obj.training.test_interval} users")
    print(f"Negative warmup: exp0~{config_obj.negative.warmup_experiences-1}")

    start_time = time.time()

    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):

        # Train on experience
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])

        # Remove negatives after warmup
        if exp_id + 1 == config_obj.negative.warmup_experiences:
            print(f"\n=== Warmup End (exp{exp_id}) → Post-warmup (exp{exp_id+1}) ===")

            acc_pre, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Pre-purge] Accuracy: {acc_pre:.2f}%")

            purge_negatives(memory_buffer, config_obj.negative.base_id)
            trainer._update_ncm()

            acc_post, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Post-purge] Accuracy: {acc_post:.2f}%")
            print(f"========================================\n")

        # Periodic evaluation
        if (exp_id + 1) % config_obj.training.test_interval == 0 or \
           exp_id == config_obj.training.num_experiences - 1:

            print(f"\n=== Evaluation at Experience {exp_id + 1} ===")

            # Evaluate on test set
            accuracy, openset_metrics = evaluate_on_test_set(
                trainer, config_obj,
                openset_mode=openset_enabled
            )

            # Update metrics
            evaluator.update(exp_id, accuracy)
            if openset_metrics:
                evaluator.update_openset(openset_metrics)

            # Calculate continual learning metrics
            avg_acc = evaluator.get_average_accuracy()
            forgetting = evaluator.get_forgetting_measure()

            # Record results
            accuracy_record = {
                'experience': exp_id + 1,
                'accuracy': accuracy,
                'average_accuracy': avg_acc,
                'forgetting': forgetting
            }

            if openset_metrics:
                accuracy_record.update(openset_metrics)
                training_history['openset_metrics'].append(openset_metrics)

            training_history['accuracies'].append(accuracy_record)

            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            print(f"Forgetting Measure: {forgetting:.2f}%")
            print(f"Memory Buffer Size: {len(memory_buffer)}")

            # Print open-set metrics
            if openset_metrics:
                print(f"\nOpen-set Metrics:")
                if 'TAR' in openset_metrics:
                    print(f"   TAR: {openset_metrics['TAR']:.3f}, FRR: {openset_metrics['FRR']:.3f}")
                if 'TRR_unknown' in openset_metrics:
                    print(f"   TRR: {openset_metrics['TRR_unknown']:.3f}, FAR: {openset_metrics['FAR_unknown']:.3f}")
                print(f"   τ_s: {openset_metrics.get('tau_s', 0):.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                results_dir,
                f'checkpoint_exp_{exp_id + 1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)

        # Progress tracking
        if (exp_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (exp_id + 1) * (config_obj.training.num_experiences - exp_id - 1)
            print(f"Progress: {exp_id + 1}/{config_obj.training.num_experiences} "
                  f"({100 * (exp_id + 1) / config_obj.training.num_experiences:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # 9. Save final results
    print("\n=== Saving Final Results ===")

    # Save training history
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)

    # Final statistics
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_acc = final_result.get('accuracy', 0)
    final_avg_acc = final_result.get('average_accuracy', 0)
    final_forget = final_result.get('forgetting', 0)

    print(f"\n=== Final Results ===")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Final Average Accuracy: {final_avg_acc:.2f}%")
    print(f"Final Forgetting Measure: {final_forget:.2f}%")

    # Final open-set metrics
    if openset_enabled and 'TAR' in final_result:
        print(f"\nFinal Open-set Performance:")
        print(f"   TAR: {final_result.get('TAR', 0):.3f}")
        print(f"   TRR (Unknown): {final_result.get('TRR_unknown', 0):.3f}")
        print(f"   FAR (Unknown): {final_result.get('FAR_unknown', 0):.3f}")
        print(f"   Final τ_s: {final_result.get('tau_s', 0):.4f}")

    print(f"\nTotal Training Time: {(time.time() - start_time)/60:.1f} minutes")

    # Save summary
    summary = {
        'config': args.config,
        'num_experiences': config_obj.training.num_experiences,
        'memory_size': config_obj.training.memory_size,
        'final_accuracy': final_acc,
        'final_average_accuracy': final_avg_acc,
        'final_forgetting': final_forget,
        'total_time_minutes': (time.time() - start_time) / 60,
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed
    }

    if openset_enabled and final_result:
        summary['final_openset'] = {
            'TAR': final_result.get('TAR', 0),
            'TRR_unknown': final_result.get('TRR_unknown', 0),
            'FAR_unknown': final_result.get('FAR_unknown', 0),
            'tau_s': final_result.get('tau_s', 0)
        }

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {results_dir}")

    print("\n" + "="*60)
    print(" COCONUT Training Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCONUT Training')
    parser.add_argument('--config', type=str, default='config/config_clean.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    main(args)