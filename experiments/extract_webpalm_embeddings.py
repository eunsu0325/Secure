"""WebPalm MFN-112 embedding extraction.

Loads a trained MFN ArcFace checkpoint and produces L2-normalized 512-d
embeddings for every successful WebPalm ROI image. The output NPZ is
consumed downstream by:
  - experiments/webpalm_dedup.py     (nearest-neighbor screen vs in-pool)
  - experiments/exp1s_orchestrator.py (Exp1-S security stress; Phase 3+)

Backbone choice (per plan §V2): use the **Tongji-trained MFN** as the
unified embedding backbone for WebPalm. Rationale: Tongji has the largest
train_backbone identity pool, giving the most general palm embedding
across the three in-pool datasets.

Resolution (per plan §C6): WebPalm ROIs are extracted at 224x224 (RegPalm-
compatible) but resized to 112x112 here for backbone forward pass to match
the unified normalized-resolution protocol.

Output NPZ schema:
  embeddings    [N, 512] float32, L2-normalized
  filename      [N]      str  (WebPalm ID = ROI filename stem without "_roi")
  split         [N]      str  (calibration / dev / test / unused)
  hand          [N]      str  (L / R / U)
  method        [N]      str  (landmark / landmark_low_conf / skin_fallback)

Usage:
    python experiments/extract_webpalm_embeddings.py \
        --roi-dir   ~/research_data/webpalm/roi_224/224 \
        --splits    ~/research_data/webpalm/splits/splits.json \
        --checkpoint experiments/generated/tongji_full/mfn_arcface_tongji_scratch_112.pt \
        --output    ~/research_data/webpalm/embeddings_mfn112.npz \
        --image-size 112
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.backbones.mobilefacenet import MobileFaceNet


class WebPalmROIDataset(Dataset):
    def __init__(self, paths: List[Path], image_size: int = 112) -> None:
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), str(p)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roi-dir", type=Path, required=True,
                   help="Directory containing per-image ROI subdirs "
                        "(e.g. ~/research_data/webpalm/roi_224/224)")
    p.add_argument("--splits", type=Path, required=True,
                   help="splits.json from split_webpalm.py")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device_str = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_str)
    print(f"[device] {device}")

    # Load splits.json: list of {filename, split, hand, method}
    splits_data = json.loads(args.splits.read_text())
    fname_to_meta: Dict[str, Dict[str, str]] = {
        r["filename"]: r for r in splits_data
    }

    # Find ROI files. Naming: <roi_dir>/<user_id>/<user_id>_roi.jpg
    roi_paths: List[Path] = []
    for sub in args.roi_dir.iterdir():
        if not sub.is_dir():
            continue
        candidates = list(sub.glob("*_roi.jpg")) + list(sub.glob("*_roi.JPG"))
        for c in candidates:
            stem = c.stem.replace("_roi", "")
            if stem in fname_to_meta:
                roi_paths.append(c)
    print(f"[scan] {len(roi_paths)} ROI files matched against splits")

    # Load backbone
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    embedding_dim = int(ckpt.get("embedding_dim", 512))
    backbone = MobileFaceNet(embedding_dim=embedding_dim).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()
    print(f"[ckpt] arch=mfn embedding_dim={embedding_dim}")

    # Sort paths for deterministic order
    roi_paths.sort()

    ds = WebPalmROIDataset(roi_paths, image_size=args.image_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    n = len(roi_paths)
    embs = np.zeros((n, embedding_dim), dtype=np.float32)
    fnames = np.empty((n,), dtype=object)
    splits = np.empty((n,), dtype=object)
    hands = np.empty((n,), dtype=object)
    methods = np.empty((n,), dtype=object)

    cursor = 0
    with torch.no_grad():
        for batch in loader:
            imgs, paths = batch
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone(imgs)
            feats = F.normalize(feats.float(), p=2, dim=1)
            bs = feats.shape[0]
            embs[cursor:cursor + bs] = feats.cpu().numpy()
            for i in range(bs):
                idx = cursor + i
                p_ = Path(paths[i])
                stem = p_.stem.replace("_roi", "")
                meta = fname_to_meta[stem]
                fnames[idx] = stem
                splits[idx] = meta["split"]
                hands[idx] = meta.get("hand", "U")
                methods[idx] = meta.get("method", "")
            cursor += bs
            if cursor % (args.batch_size * 50) == 0:
                print(f"[progress] {cursor}/{n}")

    norms = np.linalg.norm(embs, axis=1)
    print(f"[emb] L2-norm range: {norms.min():.5f} - {norms.max():.5f}")
    if not (norms > 0.99).all():
        print("WARNING: some embeddings not unit norm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embs,
        filename=fnames,
        split=splits,
        hand=hands,
        method=methods,
    )
    print(f"[save] {args.output} ({embs.shape})")

    from collections import Counter
    print(f"[counts] split: {dict(Counter(splits.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
