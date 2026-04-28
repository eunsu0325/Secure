"""Common infrastructure for Exp1 dataset builders and runner.

Modules:
    dataset_manifest -- canonical manifest writers (all.txt, identity_split.json,
        sample_split.json, metadata.json).
    split_validation -- validate_manifest(out_dir): all guards a builder must
        pass before its outputs feed exp1_run.py.
    config_factory   -- write_configs(...) emits per-(dataset, backbone) YAMLs.
    result_summary   -- write_summary(output_dir) writes summary.json after a
        run finishes; gates 5%-FPIR computation on existing artifacts.

Design contracts shared by every builder:
  * base_ids and external_ids are sorted; future_ids preserves the seeded
    shuffle order produced at selection time. The shuffle order is also
    recorded in metadata.future_subject_order so split_validation can verify
    it directly.
  * Inside each enroll/dev/test list, paths are sorted by parsed numeric
    repetition_idx (dataset-specific), not raw filename string.
  * Every JSON file is written via the dataset_manifest._dump_json helper:
    json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False) plus a
    trailing newline. List values (including future_ids) are preserved.
"""

from experiments.exp1_common.dataset_manifest import (  # noqa: F401
    IMAGE_EXTS,
    is_image_file,
    write_all_txt,
    write_identity_split,
    write_sample_split,
    write_metadata,
)
from experiments.exp1_common.split_validation import (  # noqa: F401
    ManifestValidationError,
    validate_manifest,
)
from experiments.exp1_common.config_factory import write_configs  # noqa: F401
from experiments.exp1_common.result_summary import write_summary  # noqa: F401
