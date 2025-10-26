"""
Quick tests for dataset.py (OASIS PNG layout).

What it verifies:
1) __len__/__getitem__: shapes & dtypes are correct
   - image: (1, H, W) float32 (z-score normalized)
   - mask:  (H, W)    int64    with values in [0, num_classes-1]
2) Pairing: each case ID resolves to existing image/mask paths
3) DataLoader: batches shape to (B, 1, H, W) and (B, H, W)
4) Augmentation: preserves shape & dtype
"""
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image  # only used for a raw uniqueness check if needed

from dataset import OasisSeg2D


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data",
                    help="OASIS root containing keras_png_* and keras_png_slices_seg_*")
    ap.add_argument("--list_file", type=str, default="splits/train.txt",
                    help="Split list with case IDs (e.g., splits/train.txt)")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--remap_labels", type=str, default="splits/remap_labels_oasis_png.json",
                    help="JSON mapping (0,85,170,255)->(0,1,2,3). Leave as non-existing to skip.")
    ap.add_argument("--n_samples", type=int, default=3, help="How many samples to spot-check")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    remap = args.remap_labels if Path(args.remap_labels).exists() else None
    print(f"[config] data_dir={args.data_dir}")
    print(f"[config] list_file={args.list_file}")
    print(f"[config] num_classes={args.num_classes}")
    print(f"[config] remap_labels={remap}")

    # ---- Construct dataset (no augmentation for deterministic checks) ----
    ds = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.list_file,
        num_classes=args.num_classes,
        augment=False,
        remap_labels_json=remap
    )
    print(f"[dataset] length = {len(ds)} samples")

    # ---- Sample-level checks ----
    n = min(args.n_samples, len(ds))
    indices = random.sample(range(len(ds)), k=n) if len(ds) > n else list(range(n))

    for i, idx in enumerate(indices, 1):
        img, mask, cid = ds[idx]
        # Basic prints
        print(f"\n[sample {i}/{n}] id={cid}")
        print(f"  image: shape={tuple(img.shape)} dtype={img.dtype} min={float(img.min()):.3f} max={float(img.max()):.3f}")
        print(f"  mask : shape={tuple(mask.shape)} dtype={mask.dtype} uniq={mask.unique().tolist()}")

        # Shape/dtype assertions
        assert img.dtype == torch.float32, "Image must be float32"
        assert mask.dtype == torch.int64,  "Mask must be int64 (long)"
        assert img.ndim == 3 and img.shape[0] == 1, "Image must be (1,H,W)"
        H, W = img.shape[-2], img.shape[-1]
        assert mask.ndim == 2 and mask.shape == (H, W), "Mask must be (H,W) matching image size"

        # Label range assertions (after remap)
        mi, ma = int(mask.min()), int(mask.max())
        assert mi >= 0 and ma < args.num_classes, f"Mask labels must be in [0,{args.num_classes-1}] (got [{mi},{ma}])"

        # Normalization sanity: std > 0, mean roughly near 0 (not strict)
        mean = float(img.mean())
        std  = float(img.std())
        print(f"  norm : mean={mean:.3f} std={std:.3f}")
        assert std > 0, "Image std should be > 0 after normalization"

        # Path pairing exists
        img_p, msk_p = ds.index[cid]
        assert img_p.exists() and msk_p.exists(), "Paired image/mask paths must exist on disk"
        # Optional: raw mask uniques (before remap) – useful if you’re curious
        # raw_vals = np.unique(np.array(Image.open(msk_p)))
        # print(f"  raw mask unique values on disk: {raw_vals[:8]}...")

    # ---- DataLoader batch shape check ----
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    imgs, masks, ids = batch
    print(f"\n[loader] batch shapes: images={tuple(imgs.shape)} masks={tuple(masks.shape)} ids={len(ids)}")
    assert imgs.ndim == 4 and imgs.shape[1] == 1, "Batched images must be (B,1,H,W)"
    assert masks.ndim == 3 and masks.shape[0] == imgs.shape[0], "Batched masks must be (B,H,W)"
    assert imgs.dtype == torch.float32 and masks.dtype == torch.int64, "Batch dtypes must be float32/int64"

    # ---- Augmentation smoke test (preserves shape/dtype) ----
    ds_aug = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.list_file,
        num_classes=args.num_classes,
        augment=True,
        remap_labels_json=remap
    )
    img2, mask2, _ = ds_aug[indices[0] if indices else 0]
    assert img2.shape == imgs.shape[1:], "Augmented sample must keep (1,H,W) shape"
    assert mask2.shape == masks.shape[1:], "Augmented sample must keep (H,W) shape"
    assert img2.dtype == torch.float32 and mask2.dtype == torch.int64, "Augmented dtypes must be preserved"

    print("\n✅ dataset.py tests passed.")


if __name__ == "__main__":
    main()