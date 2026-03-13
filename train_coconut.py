#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCONUT Training Script with Open-set Support
CCNet + ProxyAnchor + SupCon for Continual Learning
TTA Support Included
"""

import os
import argparse
import time
import numpy as np
from collections import defaultdict
import json
import random

import torch
from torch.utils.data import Subset

# Project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ConfigParser
from coconut.models import PretrainedLoader

# 시각화 관련 (t-SNE 디버깅용)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn.functional as F

from coconut.models import ccnet
from coconut.data import BaseVeinDataset, get_scr_transforms, ExperienceStream

# COCONUT 모듈 import
from coconut import (
    ClassBalancedBuffer,
    NCMClassifier,
    COCONUTTrainer,
    ContinualLearningEvaluator
)
from coconut.memory import HerdingBuffer

from coconut.openset import (
    predict_batch,
    predict_batch_tta,
    load_paths_labels_from_txt
)



@torch.no_grad()
def plot_tsne_from_memory(trainer,
                          save_path,
                          per_class=150,
                          space='fe',
                          perplexity=30,
                          max_points=5000,
                          seed=42):
    """메모리 버퍼의 임베딩을 t-SNE로 시각화 (디버깅용)"""
    model = trainer.model
    model.eval()
    device = trainer.device

    paths, labels = trainer.memory_buffer.get_all_data()
    if len(paths) == 0:
        print("[t-SNE] Memory buffer is empty.")
        return

    real_data = list(zip(paths, labels))

    if len(real_data) == 0:
        print("[t-SNE] No real users in buffer.")
        return

    from collections import defaultdict
    by_cls = defaultdict(list)
    for p, y in real_data:
        by_cls[int(y)].append(p)

    rng = random.Random(seed)
    sampled = []
    for c, lst in by_cls.items():
        k = min(per_class, len(lst))
        sampled += [(p, c) for p in rng.sample(lst, k)]

    if len(sampled) > max_points:
        sampled = rng.sample(sampled, max_points)

    print(f"[t-SNE] Processing {len(sampled)} samples from {len(by_cls)} classes")

    transform = trainer.test_transform
    ch = trainer.config.dataset.channels

    batch_size = 128
    features_list = []
    labels_list = []

    for i in range(0, len(sampled), batch_size):
        batch = sampled[i:i+batch_size]

        imgs = []
        for path, _ in batch:
            with Image.open(path) as img:
                img = img.convert('L' if ch == 1 else 'RGB')
                imgs.append(transform(img))

        imgs = torch.stack(imgs).to(device)
        features = model(imgs)

        if space == 'z' and hasattr(model, 'projection_head'):
            features = F.normalize(model.projection_head(features), dim=-1)

        features_list.append(features.cpu())
        labels_list.extend([y for _, y in batch])

    X = torch.cat(features_list, dim=0).numpy()
    Y = np.array(labels_list)

    perplexity = min(perplexity, max(5, len(Y)//4))

    print(f"[t-SNE] Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1500,
        metric='euclidean',
        init='pca',
        learning_rate='auto',
        random_state=seed,
        verbose=0
    )

    Z = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    n_classes = len(set(Y))
    cmap = cm.get_cmap('tab20' if n_classes <= 20 else 'hsv', n_classes)

    for idx, c in enumerate(sorted(set(Y))):
        mask = (Y == c)
        color = cmap(idx)
        plt.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[color],
            s=15,
            alpha=0.7,
            edgecolors='none',
            label=f'User {c}' if n_classes <= 10 else None
        )

    plt.title(f't-SNE Visualization ({space.upper()}) | {n_classes} classes, {len(Y)} points',
              fontsize=14, fontweight='bold')
    plt.axis('off')

    if n_classes <= 10:
        plt.legend(loc='best', markerscale=2)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[t-SNE] Saved to {save_path}")


def main(args):
    """메인 실행 함수"""

    print("\n" + "="*60)
    print("[COCONUT] COCONUT Training Starting")
    print("   CCNet + ProxyAnchor + SupCon")
    print("="*60 + "\n")

    # 1. Configuration 로드 (seed 설정 전에)
    config = ConfigParser(args.config)
    config_obj = config.get_config()
    print(f"Using config: {args.config}")
    print(config)

    # 2. Seed 결정: config에 있으면 사용, 없으면 args.seed 사용
    # args.seed가 기본값(42)이 아니면 override
    if args.seed != 42:  # Command line에서 명시적으로 설정한 경우
        seed = args.seed
        print(f"Using seed from command line: {seed}")
    elif hasattr(config_obj.training, 'seed'):
        seed = config_obj.training.seed
        print(f"Using seed from config: {seed}")
    else:
        seed = args.seed  # 기본값 42 사용
        print(f"Using default seed: {seed}")

    # 3. Set environment variables for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # 4. Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5. Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

    print(f"🔒 Reproducibility enabled with seed={seed}")

    # 6. Config에 최종 seed 저장 (COCONUTTrainer가 사용)
    config_obj.training.seed = seed

    # 오픈셋 모드 확인
    openset_enabled = hasattr(config_obj, 'openset') and config_obj.openset.enabled
    if openset_enabled:
        print("\n========== OPEN-SET MODE ENABLED ==========")
        print(f"   Warmup users: {config_obj.openset.warmup_users}")
        print(f"   Initial tau: {config_obj.openset.initial_tau}")

        # TTA 설정 (시작 시 한 번만 자세히 출력)
        if config_obj.openset.tta_n_views > 1:
            print(f"   TTA Configuration:")
            print(f"      Views: {config_obj.openset.tta_n_views} (+original: {config_obj.openset.tta_include_original})")
            print(f"      Type-specific repeats: G={config_obj.openset.tta_n_repeats_genuine}, "
                  f"B={config_obj.openset.tta_n_repeats_between}")
            print(f"      Aggregation: {config_obj.openset.tta_aggregation} (repeat: {config_obj.openset.tta_repeat_aggregation})")

        # Threshold 모드 표시
        print(f"   Threshold mode: FAR Target ({config_obj.openset.target_far*100:.1f}%)")

        print("=========================================\n")

    # ProxyAnchorLoss 설정 확인
    if hasattr(config_obj.training, 'use_proxy_anchor') and config_obj.training.use_proxy_anchor:
        print("\n========== PROXY ANCHOR LOSS ENABLED ==========")
        print(f"   Margin (δ): {config_obj.training.proxy_margin}")
        print(f"   Alpha (α): {config_obj.training.proxy_alpha}")
        print(f"   LR Ratio: {config_obj.training.proxy_lr_ratio}x")
        print(f"   Lambda (fixed): {config_obj.training.proxy_lambda}")
        print("===============================================\n")

    # GPU 설정
    device = torch.device(
        f"cuda:{config_obj.training.gpu_ids}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print(f'Device: {device}')

    # 2. 결과 저장 디렉토리 생성
    results_dir = os.path.join(config_obj.training.results_path, 'coconut_results')
    os.makedirs(results_dir, exist_ok=True)

    # 3. 데이터 스트림 초기화
    print("\n=== Initializing Data Stream ===")
    data_stream = ExperienceStream(
        train_file=str(config_obj.dataset.enroll_file),        # 등록 대상 사용자 파일
        negative_file=str(config_obj.dataset.xdomain_file),   # 크로스도메인 파일 (평가 전용)
        num_negative_classes=config_obj.dataset.num_xdomain_classes
    )

    stats = data_stream.get_statistics()
    print(f"Total users: {stats['num_users']}")
    print(f"Samples per user: {stats['samples_per_user']}")
    print(f"Negative samples: {stats['negative_samples']}")

    # 4. 모델 및 컴포넌트 초기화
    print("\n=== Initializing Model and Components ===")

    # CCNet 모델
    model = ccnet(
        weight=config_obj.model.competition_weight,
        use_projection=config_obj.model.use_projection,
        projection_dim=config_obj.model.projection_dim
    )

    # 프로젝션 헤드 설정 출력
    if config_obj.model.use_projection:
        print(f"Projection Head Configuration:")
        print(f"   Enabled: True")
        print(f"   Dimension: 6144 -> 2048 -> {config_obj.model.projection_dim}")
        print(f"   Structure: 2 layers with LayerNorm")
        print(f"   Training: Uses projection ({config_obj.model.projection_dim}D)")
        print(f"   NCM/Eval: Uses original features (6144D)")
    else:
        print(f"Projection Head: Disabled (using raw 6144D features)")

    # 사전훈련 가중치는 COCONUTTrainer에서 로드함
    # (중복 로딩 방지)
    model = model.to(device)

    # COCONUT 컴포넌트 사용
    # NCM Classifier
    ncm_classifier = NCMClassifier(normalize=True).to(device)

    # Memory Buffer (Herding or Random)
    if hasattr(config_obj.training, 'use_herding') and config_obj.training.use_herding:
        print("\n========== HERDING BUFFER ENABLED ==========")
        print(f"   Max samples per class: {config_obj.training.max_samples_per_class}")
        print(f"   Drift threshold: {config_obj.training.drift_threshold}")
        print("   Using iCaRL-inspired representative sampling")
        print("===============================================\n")

        # Transform for feature extraction
        transform = get_scr_transforms(
            train=False,
            imside=config_obj.dataset.height,
            channels=config_obj.dataset.channels
        )

        memory_buffer = HerdingBuffer(
            max_size=config_obj.training.memory_size,
            model=model,
            device=device,
            transform=transform,
            min_samples_per_class=config_obj.training.min_samples_per_class,
            max_samples_per_class=config_obj.training.max_samples_per_class,
            use_projection=config_obj.model.use_projection,
            channels=config_obj.dataset.channels
        )
        # drift_threshold 설정
        memory_buffer.drift_threshold = config_obj.training.drift_threshold
    else:
        print("Using standard Class-Balanced Buffer with random sampling")
        memory_buffer = ClassBalancedBuffer(
            max_size=config_obj.training.memory_size,
            min_samples_per_class=config_obj.training.min_samples_per_class
        )

    # COCONUT Trainer
    trainer = COCONUTTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config_obj,
        device=device
    )

    print("\nNCM starts empty (no NegRef in training)")
    print(f"Initial buffer size: {len(memory_buffer)}")

    # 6. 평가자 초기화 (새로운 per-user 평가 시스템)
    evaluator = ContinualLearningEvaluator(
        config=config_obj,
        test_file=str(config_obj.dataset.eval_probe_file)  # BWT/forgetting 측정용 독립 probe
    )

    # 7. 학습 결과 저장용
    training_history = {
        'losses': [],
        'accuracies': [],
        'forgetting_measures': [],
        'memory_sizes': [],
        'negative_removal_history': [],
        'openset_metrics': [],
        'pretrained_used': config_obj.model.use_pretrained,
        'pretrained_path': str(config_obj.model.pretrained_path) if config_obj.model.pretrained_path else None,
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed,
        'config_path': args.config,
        'trainer_type': 'COCONUTTrainer'
    }

    # 8. Continual Learning 시작
    print("\n=== Starting Continual Learning ===")
    print(f"Total experiences: {config_obj.training.num_experiences}")
    print(f"Evaluation interval: every {config_obj.training.test_interval} users")
    print(f"Learning rate: {config_obj.training.learning_rate}")
    print(f"Memory batch size: {config_obj.training.memory_batch_size}")
    print(f"Temperature: {config_obj.training.temperature}")
    print(f"Random seed: {config_obj.training.seed}")

    start_time = time.time()

    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):

        # Register user in evaluator
        evaluator.register_user(user_id)

        # Experience 학습
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])

        # 평가 주기 확인
        if (exp_id + 1) % config_obj.training.test_interval == 0 or exp_id == config_obj.training.num_experiences - 1:

            print(f"\n=== Evaluation at Experience {exp_id + 1} ===")

            # t-SNE 시각화 (디버깅용)
            if (exp_id + 1) % 10 == 0 or exp_id == config_obj.training.num_experiences - 1:
                tsne_dir = os.path.join(results_dir, "tsne")
                os.makedirs(tsne_dir, exist_ok=True)

                tsne_path = os.path.join(tsne_dir, f"tsne_exp_{exp_id+1:03d}.png")
                plot_tsne_from_memory(
                    trainer=trainer,
                    save_path=tsne_path,
                    per_class=150,
                    space='fe',
                    perplexity=30,
                    max_points=5000,
                    seed=seed
                )

                if config_obj.model.use_projection:
                    tsne_path_z = os.path.join(tsne_dir, f"tsne_z_exp_{exp_id+1:03d}.png")
                    plot_tsne_from_memory(
                        trainer=trainer,
                        save_path=tsne_path_z,
                        per_class=150,
                        space='z',
                        perplexity=30,
                        max_points=5000,
                        seed=seed
                    )

            # 모든 사용자 개별 평가 (새로운 방식)
            curves_dir = os.path.join(results_dir, "evaluation_curves", f"exp_{exp_id+1:03d}")
            report = evaluator.evaluate_all_users(
                trainer=trainer,
                experience_id=exp_id + 1,
                use_tta=trainer.use_tta if hasattr(trainer, 'use_tta') else False,
                save_curves=True,
                curves_dir=curves_dir
            )

            avg_performance = evaluator.get_average_performance('tar_001')
            forgetting = evaluator.get_forgetting_measure('tar_001')
            bwt = evaluator.get_bwt('tar_001')

            # BWT를 trainer.evaluation_history의 마지막 항목에 병합 (eval_curve.csv에 포함되도록)
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                trainer.evaluation_history[-1]['bwt'] = bwt
                trainer.evaluation_history[-1]['mean_forgetting'] = forgetting

            # 기록 저장
            accuracy_record = {
                'experience': exp_id + 1,
                'average_tar_001': avg_performance,
                'forgetting': forgetting,
                'bwt': bwt,
                'evaluated_users': report['evaluated_users'],
                'mean_forgetting': report['overall']['mean_forgetting'],
                'std_forgetting': report['overall']['std_forgetting'],
                'bwt_overall': report['overall'].get('bwt', 0.0)
            }

            training_history['accuracies'].append(accuracy_record)
            training_history['forgetting_reports'] = training_history.get('forgetting_reports', [])
            training_history['forgetting_reports'].append(report)

            print(f"\n📊 Summary:")
            print(f"Average TAR@1%FPIR: {avg_performance:.3f}")
            print(f"Mean Forgetting (TAR@1%FPIR): {report['overall']['mean_forgetting']:.4f}")
            print(f"Std Forgetting: {report['overall']['std_forgetting']:.4f}")
            print(f"BWT: {bwt:.4f}  (< 0 = 망각, 0 = 유지, > 0 = 전이)")
            print(f"Memory Buffer Size: {len(memory_buffer)}")

            # Herding Buffer의 경우 drift 통계 출력
            if isinstance(memory_buffer, HerdingBuffer):
                drift_stats = memory_buffer.get_drift_statistics()
                if drift_stats:
                    avg_drift = sum(drift_stats.values()) / len(drift_stats)
                    max_drift_class = max(drift_stats.items(), key=lambda x: x[1])
                    print(f"\n📈 Feature Drift Analysis:")
                    print(f"   Average drift: {avg_drift:.4f}")
                    print(f"   Max drift: Class {max_drift_class[0]} ({max_drift_class[1]:.4f})")
                    print(f"   Classes monitored: {len(drift_stats)}")

            # Save evaluation results
            eval_results_dir = os.path.join(results_dir, "forgetting_analysis")
            evaluator.save_results(eval_results_dir)

            # Trainer의 오픈셋 평가 히스토리도 저장
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                training_history['trainer_openset_history'] = trainer.evaluation_history

            # 체크포인트 저장
            checkpoint_path = os.path.join(
                results_dir,
                f'checkpoint_exp_{exp_id + 1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)

        # 진행 상황 출력
        if (exp_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (exp_id + 1) * (config_obj.training.num_experiences - exp_id - 1)
            print(f"Progress: {exp_id + 1}/{config_obj.training.num_experiences} "
                  f"({100 * (exp_id + 1) / config_obj.training.num_experiences:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # 9. 최종 결과 저장
    print("\n=== Saving Final Results ===")

    # 학습 기록 저장
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)

    # t-SNE 시각화 경로 추가
    tsne_images = []
    tsne_dir = os.path.join(results_dir, "tsne")
    if os.path.exists(tsne_dir):
        tsne_images = sorted([f for f in os.listdir(tsne_dir) if f.endswith('.png')])
        if tsne_images:
            print(f"\nt-SNE visualizations saved: {len(tsne_images)} images")
            print(f"   Location: {tsne_dir}")

    # 최종 통계
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_tar_001 = final_result.get('average_tar_001', 0)
    final_forget = final_result.get('mean_forgetting', 0)
    final_forget_std = final_result.get('std_forgetting', 0)
    final_bwt = final_result.get('bwt', 0)

    print(f"\n=== Final Results ===")
    print(f"Final Average TAR@1%FPIR: {final_tar_001:.3f}")
    print(f"Final Mean Forgetting (TAR@1%FPIR): {final_forget:.4f} (±{final_forget_std:.4f})")
    print(f"Final BWT: {final_bwt:.4f}")
    print(f"Evaluated Users: {final_result.get('evaluated_users', 0)}")

    print(f"\nTotal Training Time: {(time.time() - start_time)/60:.1f} minutes")

    # Export performance matrix to CSV
    performance_csv_path = os.path.join(results_dir, "performance_matrix.csv")
    evaluator.create_performance_matrix_csv(performance_csv_path)

    # Plot forgetting curves
    forgetting_plot_path = os.path.join(results_dir, "forgetting_curves.png")
    evaluator.plot_forgetting_curves(forgetting_plot_path)

    if openset_enabled and hasattr(trainer, 'save_eval_curve'):
        eval_curve_path = os.path.join(results_dir, "eval_curve.csv")
        trainer.save_eval_curve(eval_curve_path)

        # Performance vs Users 그래프 PNG
        if hasattr(trainer, 'save_eval_curve_plot'):
            eval_plot_path = os.path.join(results_dir, "performance_vs_users.png")
            trainer.save_eval_curve_plot(eval_plot_path)

    if openset_enabled and hasattr(trainer, 'save_det_curve'):
        det_curve_path = os.path.join(results_dir, "det_curve.csv")
        trainer.save_det_curve(det_curve_path)

        # DET curve PNG — visualization.py의 plot_det_curve 사용
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            det_df = pd.read_csv(det_curve_path)
            fpir_arr = det_df['FPIR'].values.astype(float)
            fnir_arr = det_df['FNIR'].values.astype(float)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpir_arr * 100, fnir_arr * 100, 'b-', linewidth=2)
            # 운영 기준점 표시
            for fp_target in [0.1, 1.0, 5.0]:
                idx = int(np.argmin(np.abs(fpir_arr * 100 - fp_target)))
                ax.plot(fpir_arr[idx] * 100, fnir_arr[idx] * 100,
                        'ro', markersize=8,
                        label=f'FPIR={fp_target:.1f}%, FNIR={fnir_arr[idx]*100:.2f}%')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('FPIR (%)', fontsize=12)
            ax.set_ylabel('FNIR (%)', fontsize=12)
            ax.set_title('DET Curve (FPIR–FNIR Trade-off)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, which='both', alpha=0.3)
            det_plot_path = os.path.join(results_dir, "det_curve.png")
            plt.savefig(det_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[Plot] DET curve saved: {det_plot_path}")
        except Exception as e:
            print(f"WARNING: Could not generate DET curve plot: {e}")

    # 결과 요약 저장
    summary = {
        'config': args.config,
        'num_experiences': config_obj.training.num_experiences,
        'memory_size': config_obj.training.memory_size,
        'final_average_tar_001': final_tar_001,
        'final_mean_forgetting': final_forget,
        'final_std_forgetting': final_forget_std,
        'final_bwt': final_bwt,
        'evaluated_users': final_result.get('evaluated_users', 0),
        'total_time_minutes': (time.time() - start_time) / 60,
        'negative_removal_history': training_history['negative_removal_history'],
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed,
        'trainer_type': 'COCONUTTrainer'
    }

    if openset_enabled and hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
        last_eval = trainer.evaluation_history[-1]
        last_metrics = last_eval.get('metrics', {})
        summary['final_openset'] = {
            'Rank1':       last_metrics.get('Rank1', 0),
            'FNIR':        last_metrics.get('FNIR', 0),
            'FRR':         last_metrics.get('FRR', 0),
            'MisID':       last_metrics.get('MisID', 0),
            'FPIR_in':     last_metrics.get('FPIR_in', 0),
            'FPIR_xdom':   last_metrics.get('FPIR_xdom', None),
            'tau_s':       last_eval.get('tau_s', 0),
            'num_users':   last_eval.get('num_users', 0),
            'tta_enabled': last_metrics.get('tta_enabled', False)
        }

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {results_dir}")

    print("\n" + "="*60)
    print("[COCONUT] COCONUT Training Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCONUT Training (CCNet + ProxyAnchor + SupCon)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    main(args)
