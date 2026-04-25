"""
Experiment 1: Protocol Comparison — 메인 실행 스크립트 (v5)

9 conditions:
  A                  — closed-set expanding-gallery identification
  B-s1/s2/s3/sF      — size-matched static open-set sweep
  C-raw-fixed        — sequential, raw cosine, fixed τ (step 0)
  C-raw-recalib      — sequential, raw cosine, recalib τ each step
  C-snorm-fixed      — sequential, S-norm, fixed τ (step 0 S-norm space)
  C-snorm-recalib    — sequential, S-norm, recalib τ each step (appendix)

Usage:
    python experiments/exp1_run.py --config experiments/exp1_config.yaml
    python experiments/exp1_run.py --config experiments/exp1_config.yaml --pretrained_path /path/to/tongji.pth
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coconut.models.ccnet import ccnet
from coconut.models.pretrained_loader import PretrainedLoader
from coconut.data.transforms import get_scr_transforms
from coconut.openset.utils import set_seed

from exp1_protocols import (
    load_and_validate_data, split_identities, split_samples_per_id,
    extract_all_embeddings,
    build_gallery_schedule, validate_static_gallery_sizes, static_size_to_step_key,
    run_closed_set_expanding,
    run_static_open_set,
    run_sequential_raw_fixed, run_sequential_raw_recalib,
    run_sequential_snorm_fixed, run_sequential_snorm_recalib,
    sanity_check_b_vs_c_recalib,
    save_results,
)
from exp1_plotting import (
    plot_fig1_gallery_size,
    plot_fig2_sequential_curves,
    plot_fig3_score_distributions,
    plot_appendix_nye_rejection,
    plot_console_summary,
)


def load_config(config_path: str, pretrained_override: str = None) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if pretrained_override:
        cfg['model']['pretrained_path'] = pretrained_override
    return cfg


def setup_model(cfg: dict, device: torch.device):
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

    transform = get_scr_transforms(train=False,
                                   imside=cfg['dataset']['height'],
                                   channels=cfg['dataset']['channels'])
    return model, transform


def build_embedding_meta(cfg: dict, seed: int, s_cfg: dict,
                         selected_ids: list) -> dict:
    return {
        'dataset_txt_file': cfg['dataset']['txt_file'],
        'dataset_base_path': cfg['dataset'].get('base_path', ''),
        'seed': int(seed),
        'n_enroll': int(s_cfg['n_enroll']),
        'n_dev': int(s_cfg['n_dev']),
        'n_test_min': int(s_cfg.get('n_test_min', 1)),
        'selected_ids': [int(x) for x in selected_ids],
        'height': int(cfg['dataset']['height']),
        'width': int(cfg['dataset']['width']),
        'channels': int(cfg['dataset']['channels']),
        'model_architecture': cfg['model']['architecture'],
        'pretrained_path': cfg['model']['pretrained_path'],
        'competition_weight': float(cfg['model']['competition_weight']),
    }


def validate_embedding_meta(meta: dict, expected_meta: dict) -> None:
    errors = []
    scalar_keys = [
        'dataset_txt_file',
        'dataset_base_path',
        'seed',
        'n_enroll',
        'n_dev',
        'n_test_min',
        'height',
        'width',
        'channels',
        'model_architecture',
        'pretrained_path',
        'competition_weight',
    ]
    for key in scalar_keys:
        if meta.get(key) != expected_meta[key]:
            errors.append((key, meta.get(key), expected_meta[key]))

    actual_ids = meta.get('selected_ids')
    expected_ids = expected_meta['selected_ids']
    if actual_ids != expected_ids:
        actual_len = len(actual_ids) if isinstance(actual_ids, list) else None
        errors.append(('selected_ids', actual_len, len(expected_ids)))

    if errors:
        raise ValueError(
            f"Embedding reuse metadata mismatch: {errors[:10]}. "
            "Regenerate embeddings without --reuse_embeddings."
        )


def validate_reused_embeddings(embeddings: dict, sample_split: dict) -> None:
    errors = []
    expected_ids = set(sample_split.keys())
    actual_ids = set(embeddings.keys())

    for uid in sorted(expected_ids - actual_ids):
        errors.append((int(uid), 'missing_id'))
    for uid in sorted(actual_ids - expected_ids):
        errors.append((int(uid), 'unexpected_id'))

    for uid in sorted(expected_ids & actual_ids):
        parts = sample_split[uid]
        for split_name in ['enroll', 'dev', 'test']:
            if split_name not in embeddings[uid]:
                errors.append((int(uid), split_name, 'missing_split'))
                continue
            actual_count = len(embeddings[uid][split_name])
            expected_count = len(parts[split_name])
            if actual_count != expected_count:
                errors.append(
                    (int(uid), split_name, actual_count, expected_count)
                )

    if errors:
        raise ValueError(
            f"Embedding reuse mismatch: {errors[:10]}. "
            "Regenerate embeddings without --reuse_embeddings."
        )


def main():
    parser = argparse.ArgumentParser(description='Exp1: Protocol Comparison (v5)')
    parser.add_argument('--config', type=str,
                        default='experiments/exp1_config.yaml')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Override pretrained model path')
    parser.add_argument('--reuse_embeddings', action='store_true',
                        help='If set, reuse embeddings from a previous run (skip STEPs 1-5)')
    args = parser.parse_args()

    cfg = load_config(args.config, args.pretrained_path)
    seed = cfg['experiment']['seed']
    set_seed(seed)

    output_dir = PROJECT_ROOT / cfg['experiment']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_fpir = cfg['scoring']['target_fpir']
    future_batch_size = cfg['protocol_c']['future_batch_size']

    id_cfg = cfg['identity_split']
    s_cfg = cfg['sample_split']
    static_gallery_sizes = cfg['static_gallery_sizes']

    # Config validation
    validate_static_gallery_sizes(
        static_gallery_sizes,
        id_cfg['n_base'], id_cfg['n_future'], future_batch_size
    )

    print(f"[CONFIG] seed={seed}, device={device}")
    print(f"[CONFIG] output_dir={output_dir}")
    print(f"[CONFIG] static_gallery_sizes={static_gallery_sizes}")
    print(f"[CONFIG] future_batch_size={future_batch_size}")

    # ============================================================
    # STEP 1-3: Data + Splits
    # ============================================================
    print("\n" + "=" * 60)
    print(" STEP 1: Load & Validate Data")
    print("=" * 60)
    txt_file = str(PROJECT_ROOT / cfg['dataset']['txt_file'])
    base_path = str(PROJECT_ROOT / cfg['dataset']['base_path']) if cfg['dataset'].get('base_path') else ""
    id_to_paths = load_and_validate_data(txt_file, base_path)

    print("\n" + "=" * 60)
    print(" STEP 2: Identity Split")
    print("=" * 60)
    all_ids = sorted(id_to_paths.keys())
    n_min = s_cfg['n_enroll'] + s_cfg['n_dev'] + s_cfg.get('n_test_min', 1)
    usable_ids = [i for i in all_ids if len(id_to_paths[i]) >= n_min]
    print(f"[SPLIT] usable IDs (>= {n_min} samples): {len(usable_ids)}")

    identity_split = split_identities(
        usable_ids, id_cfg['n_base'], id_cfg['n_future'], id_cfg['n_external'], seed
    )

    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_results(identity_split, str(splits_dir / 'identity_split.json'))

    # Gallery schedule 빌드 + 저장
    gallery_schedule = build_gallery_schedule(
        identity_split['base_ids'],
        identity_split['future_ids'],
        future_batch_size,
    )
    save_results({k: list(v) for k, v in gallery_schedule.items()},
                 str(splits_dir / 'gallery_schedule.json'))

    print("\n" + "=" * 60)
    print(" STEP 3: Sample Split")
    print("=" * 60)
    all_selected_ids = (list(identity_split['base_ids']) +
                        list(identity_split['future_ids']) +
                        list(identity_split['external_ids']))
    sample_split = split_samples_per_id(
        id_to_paths, all_selected_ids,
        n_enroll=s_cfg['n_enroll'], n_dev=s_cfg['n_dev'], base_seed=seed
    )
    save_results({str(k): v for k, v in sample_split.items()},
                 str(splits_dir / 'sample_split.json'))

    # ============================================================
    # STEP 4-5: Model + Embeddings (재사용 가능)
    # ============================================================
    emb_dir = output_dir / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / 'all_embeddings.npz'
    emb_meta_path = emb_dir / 'embedding_meta.json'
    expected_embedding_meta = build_embedding_meta(
        cfg, seed, s_cfg, all_selected_ids
    )

    if args.reuse_embeddings and emb_path.exists():
        print("\n" + "=" * 60)
        print(" STEP 4-5: Reuse embeddings (skip model loading + forward)")
        print("=" * 60)
        if not emb_meta_path.exists():
            raise ValueError(
                f"Embedding metadata is missing for {emb_path}. "
                "Legacy embedding caches cannot be safely reused; "
                "regenerate embeddings without --reuse_embeddings."
            )
        with open(emb_meta_path, 'r') as f:
            embedding_meta = json.load(f)
        validate_embedding_meta(embedding_meta, expected_embedding_meta)

        embeddings = {}
        with np.load(str(emb_path)) as npz:
            for key in npz.files:
                # key 형식: "{uid}_{split_name}"
                uid_str, split_name = key.rsplit('_', 1)
                uid = int(uid_str)
                if uid not in embeddings:
                    embeddings[uid] = {}
                embeddings[uid][split_name] = npz[key]
        validate_reused_embeddings(embeddings, sample_split)
        print(f"[EMBEDDING] loaded from {emb_path} ({len(embeddings)} IDs)")
    else:
        print("\n" + "=" * 60)
        print(" STEP 4: Model Setup")
        print("=" * 60)
        model, transform = setup_model(cfg, device)

        print("\n" + "=" * 60)
        print(" STEP 5: Extract Embeddings")
        print("=" * 60)
        embeddings = extract_all_embeddings(
            model, sample_split, transform, device,
            channels=cfg['dataset']['channels']
        )
        all_emb_flat = {}
        for uid in embeddings:
            for split_name in embeddings[uid]:
                key = f"{uid}_{split_name}"
                all_emb_flat[key] = embeddings[uid][split_name]
        np.savez_compressed(str(emb_path), **all_emb_flat)
        save_results(expected_embedding_meta, str(emb_meta_path))
        print(f"[SAVE] embeddings saved to {emb_path}")

    # ============================================================
    # STEP 6: Run conditions
    # ============================================================
    all_condition_results = {}

    # --- A ---
    print("\n" + "=" * 60)
    print(" CONDITION A: Closed-set expanding-gallery identification")
    print("=" * 60)
    res_a = run_closed_set_expanding(embeddings, gallery_schedule, device)
    save_results(res_a, str(output_dir / 'condition_A' / 'results.json'))
    all_condition_results['A'] = res_a

    # --- B-s1/s2/s3/sF ---
    b_results = {}
    for size in static_gallery_sizes:
        step_key = static_size_to_step_key(size, id_cfg['n_base'], future_batch_size)
        gallery_ids = gallery_schedule[step_key]
        print("\n" + "=" * 60)
        print(f" CONDITION B-{size}: Static open-set (gallery={step_key})")
        print("=" * 60)
        res = run_static_open_set(
            embeddings, gallery_ids, identity_split['external_ids'],
            gallery_size_label=size, gallery_step_key=step_key,
            target_fpir=target_fpir, device=device
        )
        save_results(res, str(output_dir / f'condition_B_{size}' / 'results.json'))
        b_results[size] = res
    all_condition_results['B'] = b_results

    # --- C-raw-fixed ---
    print("\n" + "=" * 60)
    print(" CONDITION C-raw-fixed: Sequential, raw cosine, fixed τ")
    print("=" * 60)
    res_c_rf = run_sequential_raw_fixed(
        embeddings, gallery_schedule,
        identity_split['future_ids'], identity_split['external_ids'],
        target_fpir, device
    )
    # score_distributions는 크니까 따로 저장
    sd = res_c_rf.pop('score_distributions', {})
    save_results(res_c_rf, str(output_dir / 'condition_C_raw_fixed' / 'results.json'))
    save_results(sd, str(output_dir / 'condition_C_raw_fixed' / 'score_distributions.json'))
    res_c_rf['score_distributions'] = sd
    all_condition_results['C_raw_fixed'] = res_c_rf

    # --- C-raw-recalib ---
    print("\n" + "=" * 60)
    print(" CONDITION C-raw-recalib: Sequential, raw cosine, recalib τ")
    print("=" * 60)
    res_c_rr = run_sequential_raw_recalib(
        embeddings, gallery_schedule,
        identity_split['future_ids'], identity_split['external_ids'],
        target_fpir, device
    )
    sd = res_c_rr.pop('score_distributions', {})
    save_results(res_c_rr, str(output_dir / 'condition_C_raw_recalib' / 'results.json'))
    save_results(sd, str(output_dir / 'condition_C_raw_recalib' / 'score_distributions.json'))
    res_c_rr['score_distributions'] = sd
    all_condition_results['C_raw_recalib'] = res_c_rr

    # --- C-snorm-fixed ---
    print("\n" + "=" * 60)
    print(" CONDITION C-snorm-fixed: Sequential, S-norm, fixed τ")
    print("=" * 60)
    res_c_sf = run_sequential_snorm_fixed(
        embeddings, gallery_schedule,
        identity_split['future_ids'], identity_split['external_ids'],
        target_fpir, device
    )
    sd = res_c_sf.pop('score_distributions', {})
    save_results(res_c_sf, str(output_dir / 'condition_C_snorm_fixed' / 'results.json'))
    save_results(sd, str(output_dir / 'condition_C_snorm_fixed' / 'score_distributions.json'))
    res_c_sf['score_distributions'] = sd
    all_condition_results['C_snorm_fixed'] = res_c_sf

    # --- C-snorm-recalib (appendix, but always run) ---
    print("\n" + "=" * 60)
    print(" CONDITION C-snorm-recalib (appendix): Sequential, S-norm, recalib τ")
    print("=" * 60)
    res_c_sr = run_sequential_snorm_recalib(
        embeddings, gallery_schedule,
        identity_split['future_ids'], identity_split['external_ids'],
        target_fpir, device
    )
    sd = res_c_sr.pop('score_distributions', {})
    save_results(res_c_sr, str(output_dir / 'condition_C_snorm_recalib' / 'results.json'))
    save_results(sd, str(output_dir / 'condition_C_snorm_recalib' / 'score_distributions.json'))
    res_c_sr['score_distributions'] = sd
    all_condition_results['C_snorm_recalib'] = res_c_sr

    # ============================================================
    # STEP 7: Sanity check (B-sF vs C-raw-recalib-final)
    # ============================================================
    print("\n" + "=" * 60)
    print(" STEP 7: Sanity check — B-sF vs C-raw-recalib-final")
    print("=" * 60)
    sF = static_gallery_sizes[-1]
    sanity_summary = sanity_check_b_vs_c_recalib(
        b_results[sF], res_c_rr
    )
    save_results(sanity_summary, str(output_dir / 'sanity_check.json'))

    # ============================================================
    # STEP 8: Figures
    # ============================================================
    print("\n" + "=" * 60)
    print(" STEP 8: Generate Figures")
    print("=" * 60)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_gallery_size(
        b_results_by_size=b_results,
        c_raw_fixed=res_c_rf,
        c_raw_recalib=res_c_rr,
        save_path=str(fig_dir / 'fig1_gallery_size.png'),
        dataset_label=Path(cfg['dataset']['txt_file']).stem,
    )
    plot_fig2_sequential_curves(
        c_raw_fixed=res_c_rf,
        c_raw_recalib=res_c_rr,
        c_snorm_fixed=res_c_sf,
        c_snorm_recalib=res_c_sr,
        save_path=str(fig_dir / 'fig2_sequential_curves.png'),
    )
    plot_fig3_score_distributions(
        c_raw_fixed=res_c_rf,
        c_raw_recalib=res_c_rr,
        c_snorm_fixed=res_c_sf,
        save_path=str(fig_dir / 'fig3_score_distributions.png'),
    )
    plot_appendix_nye_rejection(
        c_raw_fixed=res_c_rf,
        c_raw_recalib=res_c_rr,
        save_path=str(fig_dir / 'fig_appendix_nye_rejection.png'),
    )
    plot_console_summary(all_condition_results, sanity_summary)

    # ============================================================
    # STEP 9: Full results
    # ============================================================
    full_results = {
        'config': cfg,
        'A': res_a,
        'B': {str(k): v for k, v in b_results.items()},
        'C_raw_fixed': {k: v for k, v in res_c_rf.items() if k != 'score_distributions'},
        'C_raw_recalib': {k: v for k, v in res_c_rr.items() if k != 'score_distributions'},
        'C_snorm_fixed': {k: v for k, v in res_c_sf.items() if k != 'score_distributions'},
        'C_snorm_recalib': {k: v for k, v in res_c_sr.items() if k != 'score_distributions'},
        'sanity_check': sanity_summary,
    }
    save_results(full_results, str(output_dir / 'full_results.json'))

    print("\n" + "=" * 60)
    print(" EXPERIMENT 1 COMPLETE (v5: 9 conditions)")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
