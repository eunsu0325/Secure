"""
Experiment 1: Protocol Comparison — 메인 실행 스크립트
"같은 모델을 세 평가 프로토콜에 넣어 보고,
 기존 평가가 왜 sequential enrollment 문제를 못 보여주는지 증명하는 실험"

Usage:
    python experiments/exp1_run.py --config experiments/exp1_config.yaml
    python experiments/exp1_run.py --config experiments/exp1_config.yaml --pretrained_path /path/to/tongji.pth
"""

import sys
import os
import json
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coconut.models.ccnet import ccnet
from coconut.models.pretrained_loader import PretrainedLoader
from coconut.classifiers.ncm import NCMClassifier
from coconut.data.transforms import get_scr_transforms
from coconut.openset.utils import set_seed

from exp1_protocols import (
    load_and_validate_data, split_identities, split_samples_per_id,
    extract_all_embeddings, build_prototypes, calibrate_threshold,
    run_protocol_a, run_protocol_b, run_protocol_c, save_results,
)
from exp1_plotting import (
    plot_summary_table, plot_sequential_curves, plot_score_distributions,
)


def load_config(config_path: str, pretrained_override: str = None) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if pretrained_override:
        cfg['model']['pretrained_path'] = pretrained_override
    return cfg


def setup_model(cfg: dict, device: torch.device):
    """CCNet backbone + NCMClassifier 초기화"""
    model = ccnet(weight=cfg['model']['competition_weight'])
    ckpt_path = Path(cfg['model']['pretrained_path'])
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    model = PretrainedLoader.load_ccnet_pretrained(
        model, ckpt_path, device=str(device), verbose=True
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ncm = NCMClassifier(normalize=True, score_mode='cosine')
    transform = get_scr_transforms(train=False,
                                   imside=cfg['dataset']['height'],
                                   channels=cfg['dataset']['channels'])
    return model, ncm, transform


def main():
    parser = argparse.ArgumentParser(description='Exp1: Protocol Comparison')
    parser.add_argument('--config', type=str,
                        default='experiments/exp1_config.yaml')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Override pretrained model path')
    args = parser.parse_args()

    # --- Config ---
    cfg = load_config(args.config, args.pretrained_path)
    seed = cfg['experiment']['seed']
    set_seed(seed)

    output_dir = PROJECT_ROOT / cfg['experiment']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CONFIG] seed={seed}, device={device}")
    print(f"[CONFIG] output_dir={output_dir}")

    # --- 1. 데이터 로드 ---
    print("\n" + "=" * 60)
    print(" STEP 1: Load & Validate Data")
    print("=" * 60)
    txt_file = str(PROJECT_ROOT / cfg['dataset']['txt_file'])
    base_path = str(PROJECT_ROOT / cfg['dataset']['base_path']) if cfg['dataset'].get('base_path') else ""
    id_to_paths = load_and_validate_data(txt_file, base_path)

    # --- 2. Identity Split ---
    print("\n" + "=" * 60)
    print(" STEP 2: Identity Split")
    print("=" * 60)
    all_ids = sorted(id_to_paths.keys())
    s_cfg = cfg['sample_split']
    n_min = s_cfg['n_enroll'] + s_cfg['n_dev'] + s_cfg.get('n_test_min', 1)
    usable_ids = [i for i in all_ids if len(id_to_paths[i]) >= n_min]
    print(f"[SPLIT] usable IDs (>= {n_min} samples): {len(usable_ids)}")

    id_cfg = cfg['identity_split']
    identity_split = split_identities(
        usable_ids, id_cfg['n_base'], id_cfg['n_future'], id_cfg['n_external'], seed
    )

    # split 저장
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_results(identity_split, str(splits_dir / 'identity_split.json'))

    # --- 3. Sample Split ---
    print("\n" + "=" * 60)
    print(" STEP 3: Sample Split")
    print("=" * 60)
    all_selected_ids = (identity_split['base_ids'] +
                        identity_split['future_ids'] +
                        identity_split['external_ids'])
    sample_split = split_samples_per_id(
        id_to_paths, all_selected_ids,
        n_enroll=s_cfg['n_enroll'], n_dev=s_cfg['n_dev'], base_seed=seed
    )

    # sample split 저장 (경로만 저장)
    save_results({str(k): v for k, v in sample_split.items()},
                 str(splits_dir / 'sample_split.json'))

    # --- 4. Model Setup ---
    print("\n" + "=" * 60)
    print(" STEP 4: Model Setup")
    print("=" * 60)
    model, ncm, transform = setup_model(cfg, device)

    # --- 5. Embedding 추출 ---
    print("\n" + "=" * 60)
    print(" STEP 5: Extract Embeddings")
    print("=" * 60)
    embeddings = extract_all_embeddings(
        model, sample_split, transform, device,
        channels=cfg['dataset']['channels']
    )

    # embedding 저장
    emb_dir = output_dir / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    # 전체 embedding을 하나의 dict로 저장
    all_emb_flat = {}
    for uid in embeddings:
        for split_name in embeddings[uid]:
            key = f"{uid}_{split_name}"
            all_emb_flat[key] = embeddings[uid][split_name]
    np.savez_compressed(str(emb_dir / 'all_embeddings.npz'), **all_emb_flat)
    print(f"[SAVE] embeddings saved to {emb_dir / 'all_embeddings.npz'}")

    # --- 6. Threshold Calibration ---
    print("\n" + "=" * 60)
    print(" STEP 6: Threshold Calibration")
    print("=" * 60)
    # Base IDs prototype을 먼저 설정 (calibration에 사용)
    build_prototypes(ncm, embeddings, identity_split['base_ids'], device)
    threshold = calibrate_threshold(
        ncm, embeddings, identity_split['external_ids'],
        target_fpir=cfg['scoring']['target_fpir'], device=device
    )

    # --- 7. Protocol A ---
    print("\n" + "=" * 60)
    print(" STEP 7: Protocol A (Closed-set CIL)")
    print("=" * 60)
    results_a = run_protocol_a(
        embeddings, identity_split, ncm, device,
        future_batch_size=cfg['protocol_c']['future_batch_size']
    )
    save_results(results_a, str(output_dir / 'protocol_a' / 'results.json'))

    # --- 8. Protocol B ---
    print("\n" + "=" * 60)
    print(" STEP 8: Protocol B (Static Open-set)")
    print("=" * 60)
    results_b = run_protocol_b(
        embeddings, identity_split, ncm, threshold, device
    )
    save_results(results_b, str(output_dir / 'protocol_b' / 'results.json'))

    # --- 9. Protocol C ---
    print("\n" + "=" * 60)
    print(" STEP 9: Protocol C (Sequential Enrollment)")
    print("=" * 60)
    results_c = run_protocol_c(
        embeddings, identity_split, ncm, threshold, device,
        future_batch_size=cfg['protocol_c']['future_batch_size']
    )
    # score_distributions는 크니까 별도 저장
    score_dists = results_c.pop('score_distributions', {})
    save_results(results_c, str(output_dir / 'protocol_c' / 'results.json'))
    # score distributions 복원 (plotting용)
    results_c['score_distributions'] = score_dists

    # --- 10. Figures ---
    print("\n" + "=" * 60)
    print(" STEP 10: Generate Figures")
    print("=" * 60)
    fig_dir = output_dir / 'figures'

    plot_summary_table(results_a, results_b, results_c,
                       str(fig_dir / 'fig1_protocol_comparison.png'))

    plot_sequential_curves(results_c,
                           str(fig_dir / 'fig2_sequential_curves.png'))

    plot_score_distributions(results_c,
                             str(fig_dir / 'fig3_score_distributions.png'))

    # --- 전체 결과 저장 ---
    full_results = {
        'config': cfg,
        'threshold': threshold,
        'protocol_a': results_a,
        'protocol_b': results_b,
        'protocol_c': {k: v for k, v in results_c.items()
                       if k != 'score_distributions'},
    }
    save_results(full_results, str(output_dir / 'full_results.json'))

    print("\n" + "=" * 60)
    print(" EXPERIMENT 1 COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
