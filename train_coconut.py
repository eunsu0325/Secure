#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCONUT Training Script with Open-set Support
CCNet + ProxyAnchor for Continual Learning
"""

import os
import argparse
import time
import numpy as np
from collections import defaultdict
import json
import random

import torch

# Project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ConfigParser

# 시각화 관련 (t-SNE 디버깅용)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn.functional as F

from coconut.models import ccnet
from coconut.data import get_scr_transforms, ExperienceStream

# COCONUT 모듈 import
from coconut import (
    ClassBalancedBuffer,
    NCMClassifier,
    COCONUTTrainer,
    ContinualLearningEvaluator
)
from coconut.openset import (
    predict_batch,
    load_paths_labels_from_txt,
    extract_features,
)


def _json_default(o):
    """A6: numpy-aware JSON encoder.

    stdlib json can't serialize numpy scalars (np.float32 from ForgettingTracker.get_report
    via np.mean, etc.). Coerces to native Python types. Matches the encoder pattern
    already used in trainer.py:1833 for diag_history.json.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def fit_projection_pca(trainer, enroll_paths_file: str, channels: int,
                       max_n: int = 1000, verbose: bool = True):
    """Initialise trainer.projection.proj.weight with top-k PCA components
    of pretrained CCNet features sampled from enroll_file.

    PCA is unsupervised (no labels used). Reads only from enroll_file —
    the training data — so there is no leakage with unknown_test_file
    (the final FPIR evaluation pool).

    No-op when trainer.projection is None or when fewer than 2 samples can
    be extracted.

    Returns the diagnostic dict from ProjectionHead.init_with_pca, or None
    if the fit was skipped.
    """
    if trainer.projection is None:
        return None

    paths, _ = load_paths_labels_from_txt(enroll_paths_file)
    if len(paths) == 0:
        print("[PCA-init] No paths found in enroll_file; skipping PCA init.")
        return None

    # Deterministic sampling — uses trainer's seed for reproducibility.
    if len(paths) > max_n:
        rng = np.random.RandomState(getattr(trainer, 'seed', 42))
        idx = rng.choice(len(paths), max_n, replace=False)
        paths = [paths[i] for i in idx]

    # Raw 2048-D CCNet features (this is what PCA fits).
    # IMPORTANT: trainer.model is a ProjectionWrappedModel when projection is
    # enabled, and its getFeatureCode automatically applies the projection.
    # For PCA initialisation we want the UNPROJECTED 2048-D features — those
    # are exactly what PCA needs to derive the projection basis from in the
    # first place. So we bypass the wrapper here and extract from
    # ``trainer.model.ccnet`` directly.
    raw_ccnet = trainer.model.ccnet if hasattr(trainer.model, 'ccnet') else trainer.model
    feats_np = extract_features(
        raw_ccnet, paths, trainer.test_transform, trainer.device,
        batch_size=64, channels=channels,
    )
    if feats_np is None or len(feats_np) < 2:
        print(f"[PCA-init] Only {0 if feats_np is None else len(feats_np)} features extracted; skipping.")
        return None

    feats_tensor = torch.from_numpy(feats_np).float()
    assert feats_tensor.dim() == 2 and feats_tensor.size(1) == 2048, (
        f"[PCA-init] expected raw (N, 2048) features for PCA, got {tuple(feats_tensor.shape)} "
        f"(wrapper may not have been bypassed)"
    )
    info = trainer.projection.init_with_pca(feats_tensor)

    if verbose:
        print("[PCA-init] ProjectionHead weight initialised with PCA components:")
        print(f"   n_samples              = {info['n_samples']}")
        print(f"   rank_limit (=min(N-1,D))= {info['rank_limit']}")
        print(f"   usable_components      = {info['usable_components']}")
        print(f"   retained_variance_ratio= {info['retained_variance_ratio']:.4f}")
        print(f"   top-5 singular values  = {info['singular_values_top5']}")

    return info


@torch.no_grad()
def plot_tsne_from_memory(trainer,
                          save_path,
                          per_class=150,
                          space='fe',
                          perplexity=30,
                          max_points=5000,
                          seed=42,
                          verbose=False):
    """메모리 버퍼의 임베딩을 t-SNE로 시각화 (디버깅용)"""
    model = trainer.model
    model.eval()
    device = trainer.device

    paths, labels, _ = trainer.memory_buffer.get_all_data()
    if len(paths) == 0:
        if verbose:
            print("[t-SNE] Memory buffer is empty.")
        return

    real_data = list(zip(paths, labels))

    if len(real_data) == 0:
        if verbose:
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

    if verbose:
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

        features_list.append(features.cpu())
        labels_list.extend([y for _, y in batch])

    X = torch.cat(features_list, dim=0).numpy()
    Y = np.array(labels_list)

    perplexity = min(perplexity, max(5, len(Y)//4))

    if verbose:
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

    if verbose:
        print(f"[t-SNE] Saved to {save_path}")


def plot_fpir_drift_graphs(drift_log: list, results_dir: str, verbose: bool = False):
    """
    FPIR drift 핵심 그래프 3개 생성:
      1) num_classes vs avg_tar@1%  — 클래스 증가에 따른 인식 성능 추이
      2) num_classes vs fpir_in     — 클래스 증가에 따른 미등록자 오수락률 추이
      3) num_classes vs tau         — 클래스 증가에 따른 임계값 추이

    class 수 증가 → unknown score 증가 → threshold 증가 패턴 모니터링용
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available. Skipping FPIR drift plots.")
        return

    x = [d['num_classes'] for d in drift_log]
    tar_vals = [d['avg_tar_001'] for d in drift_log]
    fpir_vals = [d['fpir_in'] for d in drift_log]
    tau_vals = [d['tau'] for d in drift_log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FPIR Drift Analysis (class count increase monitoring)',
                 fontsize=14, fontweight='bold')

    # 1) num_classes vs avg_tar@1%
    ax = axes[0]
    ax.plot(x, tar_vals, 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Number of Registered Classes', fontsize=11)
    ax.set_ylabel('Avg TAR@1%FPIR', fontsize=11)
    ax.set_title('TAR@1% vs Classes\n(1% 오인식 허용 시 수락률)', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    # 추세선
    if len(x) >= 3:
        z = np.polyfit(x, tar_vals, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'b--', alpha=0.4, linewidth=1,
                label=f'Trend: {z[0]:+.4f}/class')
        ax.legend(fontsize=9)

    # 2) num_classes vs fpir_in
    ax = axes[1]
    ax.plot(x, fpir_vals, 'r-s', markersize=4, linewidth=1.5)
    ax.set_xlabel('Number of Registered Classes', fontsize=11)
    ax.set_ylabel('FPIR_in', fontsize=11)
    ax.set_title('FPIR_in vs Classes\n(미등록자 오수락률)', fontsize=11)
    ax.set_ylim(0, max(0.1, max(fpir_vals) * 1.2) if fpir_vals else 0.1)
    ax.grid(True, alpha=0.3)
    if len(x) >= 3:
        z = np.polyfit(x, fpir_vals, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'r--', alpha=0.4, linewidth=1,
                label=f'Trend: {z[0]:+.6f}/class')
        ax.legend(fontsize=9)

    # 3) num_classes vs tau
    ax = axes[2]
    ax.plot(x, tau_vals, 'k-^', markersize=4, linewidth=1.5)
    ax.set_xlabel('Number of Registered Classes', fontsize=11)
    ax.set_ylabel('Threshold (tau)', fontsize=11)
    ax.set_title('Tau vs Classes\n(임계값 변화 추이)', fontsize=11)
    ax.grid(True, alpha=0.3)
    if len(x) >= 3:
        z = np.polyfit(x, tau_vals, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'k--', alpha=0.4, linewidth=1,
                label=f'Trend: {z[0]:+.6f}/class')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "fpir_drift_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"[Plot] FPIR drift analysis saved: {save_path}")
    else:
        print(f"  [Drift] FPIR drift graph saved: {save_path}")


def main(args):
    """메인 실행 함수"""

    # 1. Configuration 로드 (seed 설정 전에)
    config = ConfigParser(args.config)
    config_obj = config.get_config()
    verbose = getattr(config_obj.training, 'verbose', False)

    if verbose:
        print("\n" + "="*60)
        print("[COCONUT] COCONUT Training Starting")
        print("   CCNet + ProxyAnchor")
        print("="*60 + "\n")
        print(f"Using config: {args.config}")
        print(config)

    # 2. Seed 결정: config에 있으면 사용, 없으면 args.seed 사용
    if args.seed != 42:
        seed = args.seed
    elif hasattr(config_obj.training, 'seed'):
        seed = config_obj.training.seed
    else:
        seed = args.seed

    if verbose:
        print(f"Using seed: {seed}")

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

    if verbose:
        print(f"Reproducibility enabled with seed={seed}")

    # 6. Config에 최종 seed 저장 (COCONUTTrainer가 사용)
    config_obj.training.seed = seed

    # 오픈셋 모드 확인
    openset_enabled = hasattr(config_obj, 'openset') and config_obj.openset.enabled

    if verbose:
        if openset_enabled:
            print("\n========== OPEN-SET MODE ENABLED ==========")
            print(f"   Warmup users: {config_obj.openset.warmup_users}")
            print(f"   Initial tau: {config_obj.openset.initial_tau}")
            print(f"   Threshold mode: FAR Target ({config_obj.openset.target_far*100:.1f}%)")
            print("=========================================\n")

        if hasattr(config_obj.training, 'use_proxy_anchor') and config_obj.training.use_proxy_anchor:
            print("\n========== PROXY ANCHOR LOSS ENABLED ==========")
            print(f"   Margin (delta): {config_obj.training.proxy_margin}")
            print(f"   Alpha: {config_obj.training.proxy_alpha}")
            print(f"   LR Ratio: {config_obj.training.proxy_lr_ratio}x")
            print(f"   Lambda (fixed): {config_obj.training.proxy_lambda}")
            print("===============================================\n")

    # GPU 설정
    device = torch.device(
        f"cuda:{config_obj.training.gpu_ids}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )

    # 2. 결과 저장 디렉토리 생성
    results_dir = os.path.join(config_obj.training.results_path, 'coconut_results')
    os.makedirs(results_dir, exist_ok=True)

    # 3. 데이터 스트림 초기화
    if verbose:
        print("\n=== Initializing Data Stream ===")

    data_stream = ExperienceStream(
        train_file=str(config_obj.dataset.enroll_file),
        negative_file=str(config_obj.dataset.xdomain_file),
        num_negative_classes=config_obj.dataset.num_xdomain_classes
    )

    stats = data_stream.get_statistics()
    if verbose:
        print(f"Total users: {stats['num_users']}")
        print(f"Samples per user: {stats['samples_per_user']}")
        print(f"Negative samples: {stats['negative_samples']}")

    # 4. 모델 및 컴포넌트 초기화
    if verbose:
        print("\n=== Initializing Model and Components ===")

    # CCNet 모델
    model = ccnet(
        weight=config_obj.model.competition_weight
    )

    model = model.to(device)

    # NCM Classifier (cosine-only)
    ncm_classifier = NCMClassifier(normalize=True).to(device)

    # Memory Buffer
    if verbose:
        print("Using standard Class-Balanced Buffer with random sampling")
    memory_buffer = ClassBalancedBuffer(
        max_size=config_obj.training.memory_size,
        min_samples_per_class=config_obj.training.min_samples_per_class,
        fair_remainder=getattr(config_obj.training, 'fair_remainder', False)
    )

    # COCONUT Trainer
    trainer = COCONUTTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config_obj,
        device=device
    )

    # ProjectionHead PCA init (no-op if use_projection_head=False)
    # Must run AFTER trainer creation (uses trainer.model + trainer.test_transform)
    # and BEFORE training starts (so first batch already sees PCA-initialised projection).
    fit_projection_pca(
        trainer,
        enroll_paths_file=str(config_obj.dataset.enroll_file),
        channels=config_obj.dataset.channels,
        max_n=1000,
        verbose=True,  # always print — important diagnostic
    )

    if verbose:
        print("\nNCM starts empty (no NegRef in training)")
        print(f"Initial buffer size: {len(memory_buffer)}")

    # 6. 평가자 초기화
    evaluator = ContinualLearningEvaluator(
        config=config_obj,
        test_file=str(config_obj.dataset.eval_probe_file)
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

    # FPIR drift 기록용
    fpir_drift_log = []

    # --- Compact 초기화 출력 (verbose=false) ---
    if not verbose:
        # Pretrained info
        pretrained_info = ""
        if config_obj.model.use_pretrained and hasattr(model, '_pretrained_load_info'):
            info = model._pretrained_load_info
            pretrained_info = f"Pretrained: {os.path.basename(info['path'])} ({info['loaded']}/{info['total']} layers)"
        elif config_obj.model.use_pretrained:
            pretrained_info = f"Pretrained: {os.path.basename(str(config_obj.model.pretrained_path))}"
        else:
            pretrained_info = "Pretrained: OFF"

        # ProxyAnchor info
        pa_info = ""
        if hasattr(config_obj.training, 'use_proxy_anchor') and config_obj.training.use_proxy_anchor:
            pa_info = (f"ProxyAnchor: delta={config_obj.training.proxy_margin}, "
                      f"alpha={config_obj.training.proxy_alpha}, "
                      f"lambda={config_obj.training.proxy_lambda}")
        else:
            pa_info = "ProxyAnchor: OFF"

        # Open-set info
        os_info = ""
        if openset_enabled:
            os_info = f"Open-set: FAR Target {config_obj.openset.target_far*100:.1f}% | tau_init={config_obj.openset.initial_tau}"
        else:
            os_info = "Open-set: OFF"

        print(f"\n[COCONUT] CCNet + ProxyAnchor | {device} | Seed={seed}")
        print(f"  {pretrained_info}")
        print(f"  {pa_info} | LR={config_obj.training.learning_rate}")
        print(f"  {os_info}")
        print(f"  Data: {stats['num_users']} users, Memory: {config_obj.training.memory_size}, "
              f"Epochs/exp: {config_obj.training.epochs_per_experience}")

    if verbose:
        print("\n=== Starting Continual Learning ===")
        print(f"Total experiences: {config_obj.training.num_experiences}")
        print(f"Evaluation interval: every {config_obj.training.test_interval} users")
        print(f"Learning rate: {config_obj.training.learning_rate}")
        print(f"Memory batch size: {config_obj.training.memory_batch_size}")
        print(f"Random seed: {config_obj.training.seed}")

    start_time = time.time()

    # paper_minimal_outputs (P-Min flag): diagnostic Phase 2 용 — per-step PNG/CSV plot 저장,
    # per-10 t-SNE, per-10 checkpoint 모두 skip. 진단에 필요한 final CSV/JSON 만 저장.
    paper_minimal = bool(getattr(config_obj.training, 'paper_minimal_outputs', False))
    if paper_minimal:
        print('[P-Min] paper_minimal_outputs=True — skipping per-step PNGs, t-SNE, per-10 checkpoints')

    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):

        # Register user in evaluator
        evaluator.register_user(user_id)

        # Experience 학습
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])

        # 평가 주기 확인
        if (exp_id + 1) % config_obj.training.test_interval == 0 or exp_id == config_obj.training.num_experiences - 1:

            if verbose:
                print(f"\n=== Evaluation at Experience {exp_id + 1} ===")

            # t-SNE 시각화 (디버깅용) — paper_minimal 모드에선 완전 skip
            is_final_exp = (exp_id == config_obj.training.num_experiences - 1)
            do_tsne = (((exp_id + 1) % 10 == 0 or is_final_exp)
                       and not paper_minimal)
            if do_tsne:
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
                    seed=seed,
                    verbose=verbose
                )

            # 모든 사용자 개별 평가 — paper_minimal 면 PNG/CSV 저장 skip (in-memory 계산만)
            curves_dir = os.path.join(results_dir, "evaluation_curves", f"exp_{exp_id+1:03d}")
            report = evaluator.evaluate_all_users(
                trainer=trainer,
                experience_id=exp_id + 1,
                save_curves=(not paper_minimal),
                curves_dir=curves_dir
            )

            avg_performance = evaluator.get_average_performance('tar_001')
            forgetting = evaluator.get_forgetting_measure('tar_001')
            bwt = evaluator.get_bwt('tar_001')

            # BWT를 trainer.evaluation_history의 마지막 항목에 병합
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                trainer.evaluation_history[-1]['bwt'] = bwt
                trainer.evaluation_history[-1]['mean_forgetting'] = forgetting

            # FPIR drift 기록
            tau_val = getattr(trainer.ncm, 'tau_s', None)
            fpir_in = None
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                last_eval = trainer.evaluation_history[-1]
                metrics = last_eval.get('metrics', {})
                fpir_in = metrics.get('achieved_FPIR@1%', None)  # B3: canonical key

            fpir_drift_log.append({
                'experience': exp_id + 1,
                'num_classes': exp_id + 1,
                'avg_tar_001': avg_performance,
                'fpir_in': fpir_in if fpir_in is not None else 0.0,
                'tau': tau_val if tau_val is not None else 0.0,
                'forgetting': report['overall']['mean_forgetting'],
                'bwt': bwt
            })

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

            if verbose:
                print(f"\n요약:")
                print(f"Average TAR@1%FPIR (1% 오탐률에서 진짜 수락률): {avg_performance:.3f}")
                print(f"Mean Forgetting (TAR@1%FPIR 기준 망각률): {report['overall']['mean_forgetting']:.4f}")
                print(f"Std Forgetting: {report['overall']['std_forgetting']:.4f}")
                print(f"BWT: {bwt:.4f}  (< 0 = 망각, 0 = 유지, > 0 = 전이)")
                print(f"Memory Buffer Size: {len(memory_buffer)}")

            # Compact 평가 출력 (한국어 설명 포함)
            if not verbose:
                print(f"  TAR@1%={avg_performance:.3f} (1% 오탐 허용 시 정상 수락률) | "
                      f"Forget={report['overall']['mean_forgetting']:.4f} (망각률 평균) | "
                      f"BWT={bwt:.4f} (음수=망각, 양수=전이) | "
                      f"Buf={len(memory_buffer)}")

            # Save evaluation results — [P-Min]은 per-step skip하되 최종 step은 저장
            # (evaluation_history JSON은 per-step FNIR CI 등 eval_curve.csv 고정컬럼에 없는 진단의 유일한 보존처)
            do_save_results = (not paper_minimal) or is_final_exp
            if do_save_results:
                eval_results_dir = os.path.join(results_dir, "forgetting_analysis")
                evaluator.save_results(eval_results_dir)

            # Trainer의 오픈셋 평가 히스토리도 저장
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                training_history['trainer_openset_history'] = trainer.evaluation_history

            # 체크포인트 저장 — paper_minimal 면 최종만 (per-10 skip)
            save_ckpt = ((exp_id + 1) == config_obj.training.num_experiences
                         or (not paper_minimal and (exp_id + 1) % 10 == 0))
            if save_ckpt:
                checkpoint_path = os.path.join(
                    results_dir,
                    f'checkpoint_exp_{exp_id + 1}.pth'
                )
                # [P-Min] optimizer/scheduler state 제외 (재추출·figure엔 model+ncm+proxy로 충분, 용량 ~2/3↓)
                trainer.save_checkpoint(checkpoint_path, include_optimizer=not paper_minimal)

        # 진행 상황 출력
        if (exp_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (exp_id + 1) * (config_obj.training.num_experiences - exp_id - 1)
            print(f"[Progress] {exp_id + 1}/{config_obj.training.num_experiences} "
                  f"({100 * (exp_id + 1) / config_obj.training.num_experiences:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

        # A1: honor config.training.num_experiences as a hard limit on the loop.
        # Previously the loop iterated all users yielded by ExperienceStream,
        # silently ignoring this knob (verified Phase 0b: smoke with =5 ran 148).
        if exp_id + 1 >= config_obj.training.num_experiences:
            break

    # 9. 최종 결과 저장
    if verbose:
        print("\n=== Saving Final Results ===")

    # 학습 기록 저장 — [P-Min] eval_curve/performance_matrix와 중복 + forgetting_reports 누적으로
    # 커질 수 있음(우리 분석 미사용) → paper_minimal이면 skip.
    if not paper_minimal:
        history_path = os.path.join(results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4, default=_json_default)  # A6

    # FPIR drift 로그 저장 (JSON + CSV)
    if fpir_drift_log:
        if not paper_minimal:  # [P-Min] json은 csv와 동일 데이터 → skip
            drift_json_path = os.path.join(results_dir, 'fpir_drift_log.json')
            with open(drift_json_path, 'w') as f:
                json.dump(fpir_drift_log, f, indent=2, default=_json_default)  # A6

        import csv as csv_mod
        drift_csv_path = os.path.join(results_dir, 'fpir_drift_log.csv')
        with open(drift_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_mod.DictWriter(f, fieldnames=['experience', 'num_classes', 'avg_tar_001', 'fpir_in', 'tau', 'forgetting', 'bwt'])
            writer.writeheader()
            for row in fpir_drift_log:
                writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

    # t-SNE 시각화 경로 추가
    tsne_images = []
    tsne_dir = os.path.join(results_dir, "tsne")
    if os.path.exists(tsne_dir):
        tsne_images = sorted([f for f in os.listdir(tsne_dir) if f.endswith('.png')])
        if tsne_images and verbose:
            print(f"\nt-SNE visualizations saved: {len(tsne_images)} images")
            print(f"   Location: {tsne_dir}")

    # 최종 통계
    total_time = (time.time() - start_time) / 60
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_tar_001 = final_result.get('average_tar_001', 0)
    final_forget = final_result.get('mean_forgetting', 0)
    final_forget_std = final_result.get('std_forgetting', 0)
    final_bwt = final_result.get('bwt', 0)

    if verbose:
        print(f"\n=== Final Results ===")
        print(f"Final Average TAR@1%FPIR: {final_tar_001:.3f}")
        print(f"Final Mean Forgetting (TAR@1%FPIR): {final_forget:.4f} (+/-{final_forget_std:.4f})")
        print(f"Final BWT: {final_bwt:.4f}")
        print(f"Evaluated Users: {final_result.get('evaluated_users', 0)}")
        print(f"\nTotal Training Time: {total_time:.1f} minutes")

    # --- Compact 최종 결과 (한국어 설명) ---
    if not verbose:
        print(f"\n{'='*50}")
        print(f"[COCONUT] Training Complete! ({total_time:.1f} minutes)")
        print(f"  Final TAR@1% (1% 오인식 허용 시 정상 사용자 수락률): {final_tar_001:.3f}")
        print(f"  Final Forgetting (이전 사용자 성능 저하 평균): {final_forget:.4f} (+/-{final_forget_std:.4f})")
        print(f"  Final BWT (역방향 전이, 음수일수록 망각 심함): {final_bwt:.4f}")
        print(f"  Evaluated Users (평가된 사용자 수): {final_result.get('evaluated_users', 0)}")
        print(f"{'='*50}")

    # Export performance matrix to CSV
    performance_csv_path = os.path.join(results_dir, "performance_matrix.csv")
    evaluator.create_performance_matrix_csv(performance_csv_path)

    # Plot forgetting curves — [P-Min] CSV에서 재생성 가능 → skip
    if not paper_minimal:
        forgetting_plot_path = os.path.join(results_dir, "forgetting_curves.png")
        evaluator.plot_forgetting_curves(forgetting_plot_path)

    if openset_enabled and hasattr(trainer, 'save_eval_curve'):
        eval_curve_path = os.path.join(results_dir, "eval_curve.csv")
        trainer.save_eval_curve(eval_curve_path)

        if not paper_minimal and hasattr(trainer, 'save_eval_curve_plot'):
            eval_plot_path = os.path.join(results_dir, "performance_vs_users.png")
            trainer.save_eval_curve_plot(eval_plot_path)

    if openset_enabled and hasattr(trainer, 'save_det_curve'):
        det_curve_path = os.path.join(results_dir, "det_curve.csv")
        trainer.save_det_curve(det_curve_path)

        # DET curve PNG — [P-Min] csv에서 재생성 가능 → paper_minimal이면 skip
        if not paper_minimal:
            try:
                import pandas as pd
                det_df = pd.read_csv(det_curve_path)
                fpir_arr = det_df['FPIR'].values.astype(float)
                fnir_arr = det_df['FNIR'].values.astype(float)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(fpir_arr * 100, fnir_arr * 100, 'b-', linewidth=2)
                for fp_target in [0.1, 1.0, 5.0]:
                    idx = int(np.argmin(np.abs(fpir_arr * 100 - fp_target)))
                    ax.plot(fpir_arr[idx] * 100, fnir_arr[idx] * 100,
                            'ro', markersize=8,
                            label=f'FPIR={fp_target:.1f}%, FNIR={fnir_arr[idx]*100:.2f}%')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('FPIR (%)', fontsize=12)
                ax.set_ylabel('FNIR (%)', fontsize=12)
                ax.set_title('DET Curve (FPIR-FNIR Trade-off)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, which='both', alpha=0.3)
                det_plot_path = os.path.join(results_dir, "det_curve.png")
                plt.savefig(det_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                if verbose:
                    print(f"[Plot] DET curve saved: {det_plot_path}")
            except Exception as e:
                print(f"WARNING: Could not generate DET curve plot: {e}")

    # --- FPIR Drift 그래프 3개 생성 --- [P-Min] csv에서 재생성 가능 → skip
    if (not paper_minimal) and fpir_drift_log and len(fpir_drift_log) >= 2:
        plot_fpir_drift_graphs(fpir_drift_log, results_dir, verbose=verbose)

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
        'total_time_minutes': total_time,
        'negative_removal_history': training_history['negative_removal_history'],
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed,
        'trainer_type': 'COCONUTTrainer'
    }

    if openset_enabled and hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
        last_eval = trainer.evaluation_history[-1]
        last_metrics = last_eval.get('metrics', {})
        # B3: pull canonical keys directly from results dict (no legacy aliases)
        summary['final_openset'] = {
            'Rank1':       last_metrics.get('Rank1', 0),
            'FNIR':        last_metrics.get('FNIR@1%FPIR', 0),
            'FRR':         last_metrics.get('det_fail@1%', 0),
            'MisID':       last_metrics.get('id_fail@1%', 0),
            'FPIR_in':     last_metrics.get('achieved_FPIR@1%', 0),
            'FPIR_xdom':   last_metrics.get('FPIR_xdom', None),
            'tau_s':       last_eval.get('tau_s', 0),
            'num_users':   last_eval.get('num_users', 0),
        }

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=_json_default)  # A6

    if verbose:
        print(f"\nResults saved to: {results_dir}")
        print("\n" + "="*60)
        print("[COCONUT] COCONUT Training Complete!")
        print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCONUT Training (CCNet + ProxyAnchor)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    main(args)
