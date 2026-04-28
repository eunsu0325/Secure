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
    python experiments/exp1_run.py --config experiments/generated/bjtu/exp1_bjtu_repvit.yaml --checkpoint_path /path/to/repvit.pth
"""

import sys
import os
import argparse
import hashlib
import json
import shutil
import yaml
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coconut.models.model_factory import build_backbone_bundle
from coconut.openset.utils import set_seed
from experiments.exp1_common import write_summary

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
    """Backwards-compatible wrapper around build_backbone_bundle.

    Returns the (model, transform) pair for legacy callers. Use
    ``build_backbone_bundle`` directly when you also need feature_dim,
    weights_source, transform_name, etc. (e.g. for embedding_meta).
    """
    bundle = build_backbone_bundle(cfg, device, project_root=PROJECT_ROOT)
    return bundle.model, bundle.transform


def _sha256_file(path) -> str:
    """SHA-256 of a configured file path. None-safe."""
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_project_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def split_external_identities_for_calibration(
    external_ids: list,
    seed: int,
    dev_fraction: float = 0.5,
    min_dev: int = 1,
    min_test: int = 1,
    fixed_offset: int = 10007,
) -> dict:
    """Split external identities into dev (calibration) and test (evaluation).

    Returns dict with 'external_dev_ids' and 'external_test_ids' (both sorted).
    """
    external_ids_sorted = sorted(external_ids)
    if len(external_ids_sorted) < min_dev + min_test:
        raise ValueError(
            f"Need at least {min_dev + min_test} external IDs for "
            f"identity-level dev/test split, got {len(external_ids_sorted)}"
        )
    rng = np.random.RandomState(seed + fixed_offset)
    shuffled = rng.permutation(external_ids_sorted)

    n_total = len(shuffled)
    n_dev_target = int(round(n_total * dev_fraction))
    n_dev = max(min_dev, min(n_dev_target, n_total - min_test))

    external_dev_ids = sorted(shuffled[:n_dev].tolist())
    external_test_ids = sorted(shuffled[n_dev:].tolist())

    return {
        'external_dev_ids': external_dev_ids,
        'external_test_ids': external_test_ids,
        'seed': seed,
        'fixed_offset': fixed_offset,
        'dev_fraction': dev_fraction,
        'source_external_ids_count': n_total,
    }


def validate_external_eval_split(
    external_ids: list,
    external_dev_ids: list,
    external_test_ids: list,
    base_ids: list,
    future_ids: list,
    sample_split: dict,
) -> None:
    """Validate external dev/test split invariants."""
    errors = []

    # Disjoint check
    dev_set = set(external_dev_ids)
    test_set = set(external_test_ids)
    if dev_set & test_set:
        errors.append("external_dev and external_test are not disjoint")

    # Union check
    if sorted(dev_set | test_set) != sorted(external_ids):
        errors.append("union of external_dev and external_test does not equal external_ids")

    # Non-empty check
    if not dev_set:
        errors.append("external_dev_ids is empty")
    if not test_set:
        errors.append("external_test_ids is empty")

    # No overlap with base/future
    base_set = set(base_ids)
    future_set = set(future_ids)
    if dev_set & base_set:
        errors.append("external_dev_ids overlaps with base_ids")
    if dev_set & future_set:
        errors.append("external_dev_ids overlaps with future_ids")
    if test_set & base_set:
        errors.append("external_test_ids overlaps with base_ids")
    if test_set & future_set:
        errors.append("external_test_ids overlaps with future_ids")

    # Sample split coverage
    for uid in external_dev_ids:
        if uid not in sample_split:
            errors.append(f"external_dev id {uid} not in sample_split")
        elif 'dev' not in sample_split[uid] or not sample_split[uid]['dev']:
            errors.append(f"external_dev id {uid} has no dev samples")

    for uid in external_test_ids:
        if uid not in sample_split:
            errors.append(f"external_test id {uid} not in sample_split")
        elif 'test' not in sample_split[uid] or not sample_split[uid]['test']:
            errors.append(f"external_test id {uid} has no test samples")

    if errors:
        raise ValueError(
            f"External eval split validation failed: {'; '.join(errors[:5])}"
        )


def build_embedding_meta(cfg: dict, seed: int, s_cfg: dict,
                         selected_ids: list,
                         backbone_meta: dict) -> dict:
    """Build the embedding_meta payload for cache reuse validation.

    ``backbone_meta`` is a small dict produced from the model factory bundle:
    {architecture, model_name, weights_source, transform_name, feature_dim,
     input_height, input_width, channels}.
    """
    txt_file_rel = cfg['dataset']['txt_file']
    txt_path = _resolve_project_path(txt_file_rel)
    identity_split_path = _resolve_project_path(cfg['dataset'].get('precomputed_identity_split'))
    sample_split_path = _resolve_project_path(cfg['dataset'].get('precomputed_sample_split'))
    metadata_path = _resolve_project_path(cfg['dataset'].get('metadata_file'))
    return {
        'dataset_txt_file': txt_file_rel,
        'dataset_base_path': cfg['dataset'].get('base_path', ''),
        'dataset_manifest_hash': _sha256_file(txt_path),
        'identity_split_hash': _sha256_file(identity_split_path),
        'sample_split_hash': _sha256_file(sample_split_path),
        'metadata_hash': _sha256_file(metadata_path),
        'seed': int(seed),
        'n_enroll': int(s_cfg['n_enroll']),
        'n_dev': int(s_cfg['n_dev']),
        'n_test_min': int(s_cfg.get('n_test_min', 1)),
        'selected_ids': [int(x) for x in selected_ids],
        'height': int(backbone_meta['input_height']),
        'width': int(backbone_meta['input_width']),
        'channels': int(backbone_meta['channels']),
        'model_architecture': str(backbone_meta['architecture']),
        'model_name': backbone_meta.get('model_name'),
        'weights_source': str(backbone_meta['weights_source']),
        'transform_name': str(backbone_meta['transform_name']),
        'feature_dim': int(backbone_meta['feature_dim']),
        # CCNet-only; kept None for RepViT to keep schema stable.
        'competition_weight': (
            float(cfg['model']['competition_weight'])
            if cfg.get('model', {}).get('architecture') == 'ccnet'
            else None
        ),
    }


def validate_embedding_meta(meta: dict, expected_meta: dict) -> None:
    errors = []
    scalar_keys = [
        'dataset_txt_file',
        'dataset_base_path',
        'dataset_manifest_hash',
        'identity_split_hash',
        'sample_split_hash',
        'metadata_hash',
        'seed',
        'n_enroll',
        'n_dev',
        'n_test_min',
        'height',
        'width',
        'channels',
        'model_architecture',
        'model_name',
        'weights_source',
        'transform_name',
        'feature_dim',
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
            "cache is stale; regenerate embeddings without --reuse_embeddings."
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
            "cache is stale; regenerate embeddings without --reuse_embeddings."
        )


def main():
    parser = argparse.ArgumentParser(description='Exp1: Protocol Comparison (v5)')
    parser.add_argument('--config', type=str,
                        default='experiments/exp1_config.yaml')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Override pretrained model path')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Override RepViT checkpoint path (RepViT configs only)')
    parser.add_argument('--reuse_embeddings', action='store_true',
                        help='If set, reuse embeddings from a previous run (skip STEPs 1-5)')
    parser.add_argument('--strict_sanity', action='store_true',
                        help='Raise AssertionError on B-final ↔ C-raw-recalib-final '
                             'sanity failure (default: warn-and-continue).')
    parser.add_argument('--allow_random_sample_split_debug', action='store_true',
                        help='Allow random sample-split fallback for debugging only. '
                             'Paper Exp1 requires precomputed_sample_split in the config.')
    args = parser.parse_args()

    cfg = load_config(args.config, args.pretrained_path)
    if args.checkpoint_path:
        architecture = str(cfg.get('model', {}).get('architecture', '')).lower()
        if architecture != 'repvit':
            raise ValueError("--checkpoint_path is only valid for RepViT configs")
        cfg['model']['checkpoint_path'] = args.checkpoint_path
    seed = cfg['experiment']['seed']
    set_seed(seed)

    output_dir = PROJECT_ROOT / cfg['experiment']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_fpir = cfg['scoring']['target_fpir']
    run_snorm_recalib = cfg.get('scoring', {}).get('run_snorm_recalib', True) is not False
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
    print(f"[CONFIG] run_snorm_recalib={run_snorm_recalib}")

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

    precomputed_id_split_path = cfg['dataset'].get('precomputed_identity_split')
    if precomputed_id_split_path:
        full_path = Path(precomputed_id_split_path)
        if not full_path.is_absolute():
            full_path = PROJECT_ROOT / full_path
        with open(full_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        identity_split = {
            'base_ids': [int(x) for x in loaded['base_ids']],
            'future_ids': [int(x) for x in loaded['future_ids']],
            'external_ids': [int(x) for x in loaded['external_ids']],
        }
        print(f"[SPLIT] using precomputed identity split: {full_path}")
    else:
        identity_split = split_identities(
            usable_ids, id_cfg['n_base'], id_cfg['n_future'], id_cfg['n_external'], seed
        )

    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_results(identity_split, str(splits_dir / 'identity_split.json'))

    # Split external_ids into dev (calibration) and test (evaluation).
    # Compute in memory only — validation requires sample_split (loaded in STEP 3),
    # so the JSON file is persisted only after validation passes.
    external_eval_split_dict = split_external_identities_for_calibration(
        identity_split['external_ids'], seed
    )
    external_dev_ids = external_eval_split_dict['external_dev_ids']
    external_test_ids = external_eval_split_dict['external_test_ids']

    # Gallery schedule 빌드 + 저장 (always recomputed; deterministic given splits)
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
    precomputed_sample_split_path = cfg['dataset'].get('precomputed_sample_split')
    if precomputed_sample_split_path:
        full_path = Path(precomputed_sample_split_path)
        if not full_path.is_absolute():
            full_path = PROJECT_ROOT / full_path
        with open(full_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        sample_split = {}
        base_dir = ""
        if cfg['dataset'].get('base_path'):
            base_dir = str(PROJECT_ROOT / cfg['dataset']['base_path'])
        for label_str, parts in loaded.items():
            uid = int(label_str)
            sample_split[uid] = {
                'enroll': [str(p) for p in parts.get('enroll', [])],
                'dev':    [str(p) for p in parts.get('dev', [])],
                'test':   [str(p) for p in parts.get('test', [])],
            }
        # Verify the precomputed file covers exactly the selected ids.
        missing = [uid for uid in all_selected_ids if uid not in sample_split]
        extra = [uid for uid in sample_split if uid not in all_selected_ids]
        if missing or extra:
            raise ValueError(
                f"Precomputed sample_split mismatch with identity_split: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        print(f"[SPLIT] using precomputed sample split: {full_path}")
    else:
        if not args.allow_random_sample_split_debug:
            raise ValueError(
                "Paper Exp1 requires dataset.precomputed_sample_split to be set in the config. "
                "Random sample-split fallback is disabled because each dataset has a session/"
                "train/test policy that the manifest builder encodes deterministically. "
                "Use --allow_random_sample_split_debug only for debugging."
            )
        print("[SPLIT][DEBUG] random sample-split fallback engaged because --allow_random_sample_split_debug was passed")
        sample_split = split_samples_per_id(
            id_to_paths, all_selected_ids,
            n_enroll=s_cfg['n_enroll'], n_dev=s_cfg['n_dev'], base_seed=seed
        )
        save_results({"source": "random_fallback_debug", "seed": seed}, str(splits_dir / 'sample_split_source.json'))
    save_results({str(k): v for k, v in sample_split.items()},
                 str(splits_dir / 'sample_split.json'))

    # Validate external dev/test split now that sample_split is populated.
    # Defensive int-coercion: validator compares int uids against sample_split
    # keys, so any future loader change that leaves str keys must not silently
    # break the coverage check.
    sample_split_int_keys = {int(k): v for k, v in sample_split.items()}
    validate_external_eval_split(
        identity_split['external_ids'],
        external_dev_ids, external_test_ids,
        identity_split['base_ids'], identity_split['future_ids'],
        sample_split_int_keys,
    )
    # Validation passed — persist the split as authoritative.
    save_results({
        'external_dev_ids': external_dev_ids,
        'external_test_ids': external_test_ids,
        'seed': external_eval_split_dict['seed'],
        'fixed_offset': external_eval_split_dict['fixed_offset'],
        'dev_fraction': external_eval_split_dict['dev_fraction'],
        'source_external_ids_count': external_eval_split_dict['source_external_ids_count'],
    }, str(splits_dir / 'external_eval_split.json'))

    # Copy upstream metadata.json into the run output for traceability.
    metadata_file = cfg['dataset'].get('metadata_file')
    if metadata_file:
        meta_src = Path(metadata_file)
        if not meta_src.is_absolute():
            meta_src = PROJECT_ROOT / meta_src
        if meta_src.is_file():
            shutil.copy2(str(meta_src), str(output_dir / 'metadata.json'))
            print(f"[META] copied {meta_src.name} -> {output_dir / 'metadata.json'}")

    # ============================================================
    # STEP 4-5: Model + Embeddings (재사용 가능)
    # ============================================================
    emb_dir = output_dir / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / 'all_embeddings.npz'
    emb_meta_path = emb_dir / 'embedding_meta.json'

    print("\n" + "=" * 60)
    print(" STEP 4: Model Setup")
    print("=" * 60)
    bundle = build_backbone_bundle(cfg, device, project_root=PROJECT_ROOT)
    # Defensive: confirm the factory left every parameter frozen.
    assert not any(p.requires_grad for p in bundle.model.parameters()), \
        "Exp1 backbone must be frozen — found parameter with requires_grad=True"
    print(f"[MODEL] architecture={bundle.architecture} feature_dim={bundle.feature_dim} "
          f"transform={bundle.transform_name} weights_source={bundle.weights_source}")

    backbone_meta = {
        'architecture': bundle.architecture,
        'model_name': bundle.model_name,
        'weights_source': bundle.weights_source,
        'transform_name': bundle.transform_name,
        'feature_dim': bundle.feature_dim,
        'input_height': bundle.input_height,
        'input_width': bundle.input_width,
        'channels': bundle.channels,
    }
    expected_embedding_meta = build_embedding_meta(
        cfg, seed, s_cfg, all_selected_ids, backbone_meta
    )

    if args.reuse_embeddings and emb_path.exists():
        print("\n" + "=" * 60)
        print(" STEP 5: Reuse embeddings (skip forward pass)")
        print("=" * 60)
        if not emb_meta_path.exists():
            raise ValueError(
                f"Embedding metadata is missing for {emb_path}. "
                "Legacy embedding caches cannot be safely reused; "
                "cache is stale; regenerate embeddings without --reuse_embeddings."
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
        print(" STEP 5: Extract Embeddings")
        print("=" * 60)
        embeddings = extract_all_embeddings(
            bundle.model, sample_split, bundle.transform, device,
            channels=bundle.channels,
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
            embeddings, gallery_ids, external_dev_ids, external_test_ids,
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
        identity_split['future_ids'], external_dev_ids, external_test_ids,
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
        identity_split['future_ids'], external_dev_ids, external_test_ids,
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
        identity_split['future_ids'], external_dev_ids, external_test_ids,
        target_fpir, device
    )
    sd = res_c_sf.pop('score_distributions', {})
    save_results(res_c_sf, str(output_dir / 'condition_C_snorm_fixed' / 'results.json'))
    save_results(sd, str(output_dir / 'condition_C_snorm_fixed' / 'score_distributions.json'))
    res_c_sf['score_distributions'] = sd
    all_condition_results['C_snorm_fixed'] = res_c_sf

    # --- C-snorm-recalib (appendix, optional) ---
    res_c_sr = None
    if run_snorm_recalib:
        print("\n" + "=" * 60)
        print(" CONDITION C-snorm-recalib (appendix): Sequential, S-norm, recalib τ")
        print("=" * 60)
        res_c_sr = run_sequential_snorm_recalib(
            embeddings, gallery_schedule,
            identity_split['future_ids'], external_dev_ids, external_test_ids,
            target_fpir, device
        )
        sd = res_c_sr.pop('score_distributions', {})
        save_results(res_c_sr, str(output_dir / 'condition_C_snorm_recalib' / 'results.json'))
        save_results(sd, str(output_dir / 'condition_C_snorm_recalib' / 'score_distributions.json'))
        res_c_sr['score_distributions'] = sd
        all_condition_results['C_snorm_recalib'] = res_c_sr
    else:
        print("\n" + "=" * 60)
        print(" CONDITION C-snorm-recalib skipped (scoring.run_snorm_recalib=false)")
        print("=" * 60)

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

    sanity_pass = bool(sanity_summary.get('all_pass', sanity_summary.get('sanity_pass', False)))
    if not sanity_pass:
        warning_banner = "*" * 60
        print(f"\n{warning_banner}")
        print(" WARNING: B-final ↔ C-raw-recalib-final sanity check FAILED.")
        print(f" Details written to: {output_dir / 'sanity_check.json'}")
        print(warning_banner)
        if args.strict_sanity:
            raise AssertionError(
                "Sanity check failed under --strict_sanity. "
                "See sanity_check.json for diagnostic details. "
                "Figures and summary were not produced."
            )

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
        target_fpir=target_fpir,
    )
    plot_fig2_sequential_curves(
        c_raw_fixed=res_c_rf,
        c_raw_recalib=res_c_rr,
        c_snorm_fixed=res_c_sf,
        c_snorm_recalib=res_c_sr,
        save_path=str(fig_dir / 'fig2_sequential_curves.png'),
        target_fpir=target_fpir,
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
    plot_console_summary(all_condition_results, sanity_summary, target_fpir=target_fpir)

    # ============================================================
    # STEP 9: Full results
    # ============================================================
    full_results = {
        'config': cfg,
        'external_eval_split': {
            'external_dev_ids': [int(x) for x in external_dev_ids],
            'external_test_ids': [int(x) for x in external_test_ids],
            'external_dev_identity_count': len(external_dev_ids),
            'external_test_identity_count': len(external_test_ids),
            'seed': int(seed),
            'fixed_offset': int(external_eval_split_dict['fixed_offset']),
            'dev_fraction': float(external_eval_split_dict['dev_fraction']),
        },
        'A': res_a,
        'B': {str(k): v for k, v in b_results.items()},
        'C_raw_fixed': {k: v for k, v in res_c_rf.items() if k != 'score_distributions'},
        'C_raw_recalib': {k: v for k, v in res_c_rr.items() if k != 'score_distributions'},
        'C_snorm_fixed': {k: v for k, v in res_c_sf.items() if k != 'score_distributions'},
        'sanity_check': sanity_summary,
    }
    if res_c_sr is not None:
        full_results['C_snorm_recalib'] = {
            k: v for k, v in res_c_sr.items() if k != 'score_distributions'
        }
    save_results(full_results, str(output_dir / 'full_results.json'))

    # ============================================================
    # STEP 10: Flat summary.json (for downstream consumers)
    # ============================================================
    try:
        summary_path = write_summary(
            output_dir,
            dataset_name=cfg['dataset'].get('name'),
            backbone_name=cfg['model'].get('architecture'),
        )
        print(f"[SUMMARY] {summary_path}")
    except Exception as exc:
        print(f"[SUMMARY] WARNING: write_summary failed: {exc!r}")

    print("\n" + "=" * 60)
    print(" EXPERIMENT 1 COMPLETE (v5: 9 conditions)")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
