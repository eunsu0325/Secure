import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-exp1-tests"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiments.exp1_common.calibration import threshold_from_dev_scores
from experiments.exp1_common.result_summary import write_summary
from experiments.exp1_common.split_validation import (
    ManifestValidationError,
    validate_manifest,
)
from experiments.dataset_builders import build_xjtu


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _score_blob():
    known_scores = [0.5, 0.45, 0.39, 0.6]
    external_scores = [0.9, 0.95, 0.96]
    return {
        "external_dev_impostor_max_scores": [0.1, 0.2, 0.3, 0.4, 0.5],
        "known_scores": known_scores,
        "known_max_scores": list(known_scores),
        "known_pred_ids": [10, 11, 12, 13],
        "known_probe_keys": [[10, 0], [99, 0], [12, 0], [13, 0]],
        "external_scores": external_scores,
        "external_max_scores": list(external_scores),
    }


def _write_full_results_for_additional_fpirs(output_dir: Path) -> dict:
    b_blob = _score_blob()
    c_blob = _score_blob()
    full_results = {
        "config": {
            "dataset": {"name": "synthetic"},
            "model": {"architecture": "ccnet"},
            "scoring": {"target_fpir": 0.01, "additional_fpirs": [0.4]},
            "static_gallery_sizes": [2],
        },
        "A": {"steps": [{"gallery_size": 2, "rank1": 1.0}]},
        "B": {
            "2": {
                **b_blob,
                "gallery_size": 2,
                "rank1": 1.0,
                "tpir_at_target_fpir": 1.0,
                "tpir_at_1pct_fpir": 1.0,
                "achieved_external_fpir": 0.01,
            }
        },
        "C_raw_recalib": {
            "steps": [
                {
                    "step": "step_0",
                    "gallery_size": 2,
                    "rank1": 1.0,
                    "tpir_at_target_fpir": 1.0,
                    "tpir_at_1pct_fpir": 1.0,
                    "achieved_external_fpir": 0.01,
                    "threshold": 0.4,
                }
            ],
            "score_distributions": {"step_0": c_blob},
        },
        "sanity_check": {"sanity_pass": True},
    }
    _write_json(output_dir / "full_results.json", full_results)
    summary_path = write_summary(output_dir)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_threshold_from_scores_method_higher():
    assert threshold_from_dev_scores([0.1, 0.2, 0.3, 0.4, 0.5], 0.4) == 0.4


def test_additional_fpir_summary_uses_dev_known_and_external_scores(tmp_path):
    summary = _write_full_results_for_additional_fpirs(tmp_path)

    assert summary["additional_fpirs_status"] == "computed"
    assert summary["C_raw_recalib_final_tau_at_0.4_fpir"] == pytest.approx(0.4)
    assert summary["C_raw_recalib_final_tpir_at_0.4_fpir"] == pytest.approx(0.5)
    assert summary["C_raw_recalib_final_achieved_external_fpir_at_0.4_fpir"] == pytest.approx(1.0)


def test_b_final_additional_fpir_keys_are_computed(tmp_path):
    summary = _write_full_results_for_additional_fpirs(tmp_path)

    assert summary["B_final_tau_at_0.4_fpir"] == pytest.approx(0.4)
    assert summary["B_final_tpir_at_0.4_fpir"] == pytest.approx(0.5)
    assert summary["B_final_achieved_external_fpir_at_0.4_fpir"] == pytest.approx(1.0)


def test_protocol_static_result_score_aliases_match_with_external_dev_test_split():
    import torch
    from exp1_protocols import run_static_open_set

    embeddings = {
        0: {
            "enroll": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            "dev": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "test": np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        },
        1: {
            "enroll": np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "dev": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "test": np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        },
        2: {
            "enroll": np.asarray([[0.5, 0.5]], dtype=np.float32),
            "dev": np.asarray([[0.5, 0.5]], dtype=np.float32),
            "test": np.asarray([[0.5, 0.5], [0.4, 0.6]], dtype=np.float32),
        },
        3: {
            "enroll": np.asarray([[0.6, 0.4]], dtype=np.float32),
            "dev": np.asarray([[0.6, 0.4]], dtype=np.float32),
            "test": np.asarray([[0.6, 0.4], [0.7, 0.3]], dtype=np.float32),
        },
    }
    result = run_static_open_set(
        embeddings=embeddings,
        gallery_ids=[0, 1],
        external_dev_ids=[2],
        external_test_ids=[3],
        gallery_size_label=2,
        gallery_step_key="step_0",
        target_fpir=0.5,
        device=torch.device("cpu"),
    )

    assert result["known_scores"] == result["known_max_scores"]
    assert result["external_scores"] == result["external_max_scores"]


def test_sample_split_hash_mutation_invalidates_reuse_metadata(tmp_path):
    import experiments.exp1_run as exp1_run

    all_txt = tmp_path / "all.txt"
    sample_split = tmp_path / "sample_split.json"
    all_txt.write_text("a.jpg 0\n", encoding="utf-8")
    _write_json(sample_split, {"0": {"enroll": ["a.jpg"], "dev": ["b.jpg"], "test": ["c.jpg"]}})

    cfg = {
        "dataset": {
            "txt_file": str(all_txt),
            "base_path": "",
            "precomputed_sample_split": str(sample_split),
        },
        "model": {"architecture": "ccnet", "competition_weight": 0.8},
    }
    backbone_meta = {
        "input_height": 128,
        "input_width": 128,
        "channels": 1,
        "architecture": "ccnet",
        "model_name": None,
        "weights_source": "weights-a",
        "transform_name": "scr_eval_1x128",
        "feature_dim": 2048,
    }
    meta = exp1_run.build_embedding_meta(
        cfg, seed=42, s_cfg={"n_enroll": 1, "n_dev": 1, "n_test_min": 1},
        selected_ids=[0], backbone_meta=backbone_meta
    )

    _write_json(sample_split, {"0": {"enroll": ["z.jpg"], "dev": ["b.jpg"], "test": ["c.jpg"]}})
    expected = exp1_run.build_embedding_meta(
        cfg, seed=42, s_cfg={"n_enroll": 1, "n_dev": 1, "n_test_min": 1},
        selected_ids=[0], backbone_meta=backbone_meta
    )

    with pytest.raises(ValueError, match="cache is stale; regenerate embeddings without --reuse_embeddings"):
        exp1_run.validate_embedding_meta(meta, expected)


def test_split_validation_catches_sample_paths_absent_from_all_txt(tmp_path):
    out_dir = tmp_path / "manifest"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sample_split = {}
    all_records = []
    for label in range(3):
        enroll = f"data/{label}_enroll.jpg"
        dev = f"data/{label}_dev.jpg"
        test_a = f"data/{label}_test_a.jpg"
        test_b = f"data/{label}_test_b.jpg"
        for rel_path in (enroll, dev, test_a, test_b):
            (tmp_path / rel_path).write_bytes(b"x")
        sample_split[str(label)] = {
            "enroll": [enroll],
            "dev": [dev],
            "test": [test_a, test_b],
        }
        all_records.extend([(enroll, label), (dev, label), (test_a, label), (test_b, label)])

    all_records.remove(("data/2_test_b.jpg", 2))
    out_dir.mkdir()
    with open(out_dir / "all.txt", "w", encoding="utf-8") as f:
        for rel_path, label in all_records:
            f.write(f"{rel_path} {label}\n")
    _write_json(out_dir / "identity_split.json", {
        "base_ids": [0],
        "future_ids": [1],
        "external_ids": [2],
    })
    _write_json(out_dir / "sample_split.json", sample_split)
    _write_json(out_dir / "metadata.json", {
        "subject_disjoint": False,
        "future_subject_order": [1],
    })

    with pytest.raises(ManifestValidationError, match="missing_in_txt"):
        validate_manifest(out_dir, project_root=tmp_path)


def _populate_xjtu_device(root: Path, subjects):
    for illum in ("Flash", "Nature"):
        for subject in subjects:
            for hand in ("L", "R"):
                palm_dir = root / illum / f"{hand}_{subject}"
                palm_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(10):
                    (palm_dir / f"{idx:02d}.jpg").write_bytes(b"x")


def test_xjtu_modes_have_distinct_names_yamls_and_output_dirs(tmp_path):
    subjects = ["001", "002", "003"]
    huawei_root = tmp_path / "huawei"
    iphone_root = tmp_path / "iPhone"
    _populate_xjtu_device(huawei_root, subjects)
    _populate_xjtu_device(iphone_root, subjects)

    flash_out = tmp_path / "generated" / "xjtu_flash_to_nature"
    cross_out = tmp_path / "generated" / "xjtu_cross_device"
    common_args = [
        "--seed", "7",
        "--n_base_subjects", "1",
        "--n_future_subjects", "1",
        "--n_external_subjects", "1",
    ]
    assert build_xjtu.main([
        "--huawei_root", str(huawei_root),
        "--out", str(flash_out),
        "--mode", "flash_to_nature",
        *common_args,
    ]) == 0
    assert build_xjtu.main([
        "--huawei_root", str(huawei_root),
        "--iphone_root", str(iphone_root),
        "--out", str(cross_out),
        "--mode", "cross_device",
        *common_args,
    ]) == 0

    flash_meta = json.loads((flash_out / "metadata.json").read_text(encoding="utf-8"))
    cross_meta = json.loads((cross_out / "metadata.json").read_text(encoding="utf-8"))
    assert flash_meta["dataset_name"] == "xjtu_flash_to_nature"
    assert cross_meta["dataset_name"] == "xjtu_cross_device"
    assert (flash_out / "exp1_xjtu_flash_to_nature_ccnet.yaml").is_file()
    assert (cross_out / "exp1_xjtu_cross_device_repvit.yaml").is_file()
    assert flash_out != cross_out

    with pytest.raises(ValueError, match="experiments/generated/xjtu"):
        build_xjtu.main([
            "--huawei_root", str(huawei_root),
            "--out", str(tmp_path / "generated" / "xjtu"),
            "--mode", "flash_to_nature",
            *common_args,
        ])


def test_run_snorm_recalib_false_summary_and_plotting_tolerate_absence(tmp_path):
    from exp1_plotting import plot_console_summary, plot_fig2_sequential_curves

    full_results = {
        "config": {
            "dataset": {"name": "synthetic"},
            "model": {"architecture": "ccnet"},
            "scoring": {
                "target_fpir": 0.01,
                "additional_fpirs": [],
                "run_snorm_recalib": False,
            },
            "static_gallery_sizes": [2],
        },
        "A": {"steps": [{"gallery_size": 2, "rank1": 1.0}]},
        "B": {"2": {"gallery_size": 2, "rank1": 1.0, "tpir_at_target_fpir": 0.8}},
        "C_raw_fixed": {"steps": [{"step": "step_0", "gallery_size": 2, "rank1": 1.0}]},
        "C_raw_recalib": {"steps": [{"step": "step_0", "gallery_size": 2, "rank1": 1.0}]},
        "C_snorm_fixed": {"steps": [{"step": "step_0", "gallery_size": 2, "rank1": 1.0}]},
        "sanity_check": {"sanity_pass": True},
    }
    _write_json(tmp_path / "full_results.json", full_results)
    summary_path = write_summary(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["additional_fpirs_status"] == "not_requested"
    assert "C_snorm_recalib" not in full_results

    step = {
        "step": "step_0",
        "gallery_size": 2,
        "rank1": 1.0,
        "tpir_at_target_fpir": 0.8,
        "external_rejection_rate": 0.99,
        "achieved_external_fpir": 0.01,
        "threshold": 0.4,
        "not_yet_enrolled_rejection": 1.0,
        "not_yet_enrolled_probes": 10,
    }
    c_result = {"steps": [step]}
    plot_fig2_sequential_curves(
        c_raw_fixed=c_result,
        c_raw_recalib=c_result,
        c_snorm_fixed=c_result,
        c_snorm_recalib=None,
        save_path=str(tmp_path / "fig2.png"),
        include_snorm_recalib=True,
    )
    plot_console_summary(
        {
            "A": {"steps": [step]},
            "B": {2: step},
            "C_raw_fixed": c_result,
            "C_raw_recalib": c_result,
            "C_snorm_fixed": c_result,
        },
        {"sanity_pass": True},
    )
    assert (tmp_path / "fig2.png").is_file()
