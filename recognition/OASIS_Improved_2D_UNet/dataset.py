"""
Loads OASIS PNG slices with paired segmentation masks.
Handles the conversion from PNG to Tensors

Returns:
  image: (1, H, W) float32  — z-score normalized per image
  mask:  (H, W)   int64     — class IDs (e.g., 0/1/2/3)
"""

from typing import Optional, Dict, Tuple, List
from pathlib import Path
import random, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _imread_float(path: str) -> np.ndarray:
    """Grayscale PNG -> float32 array (H, W). 'F' gives float pixels."""
    return np.array(Image.open(path).convert("F"), dtype=np.float32)

def _imread_label(path: str) -> np.ndarray:
    """Label PNG -> int64 array (H, W). Values are class IDs or raw codes."""
    return np.array(Image.open(path), dtype=np.int64)


class OasisSeg2D(Dataset):
    """
    Args:
      data_dir: OASIS root (contains keras_png_* and keras_png_slices_seg_*).
      id_list_file: text file with lines like 'case_001_slice_0'.
      num_classes: number of classes incl. background.
      augment: enable random flips/rotations (training only).
      remap_labels_json: optional mapping {0:0,85:1,170:2,255:3}.
    """
    def __init__(self,
                 data_dir: str,
                 id_list_file: str,
                 num_classes: int,
                 augment: bool = True,
                 remap_labels_json: Optional[str] = None):
        self.data_dir = Path(data_dir)
        with open(id_list_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.num_classes = num_classes
        self.augment = augment

        # Optional label remap (e.g., 0/85/170/255 -> 0/1/2/3)
        self.remap: Optional[Dict[int,int]] = None
        if remap_labels_json:
            with open(remap_labels_json, "r") as f:
                self.remap = {int(k): int(v) for k, v in json.load(f).items()}

        # Verify OASIS PNG layout exists (preferred) or fallback to flat layout
        oasis_png_dirs = [
            "keras_png_slices_train", "keras_png_slices_validate", "keras_png_slices_test",
            "keras_png_slices_seg_train", "keras_png_slices_seg_validate", "keras_png_slices_seg_test",
        ]
        # Check that data directories exist
        if not all((self.data_dir / d).exists() for d in oasis_png_dirs):

            raise FileNotFoundError(
                f"Unrecognized layout in {self.data_dir}. "
                "Expected OASIS keras_png_* + keras_png_slices_seg_* OR images/ + masks/."
            )

        # Build ID -> (image_path, mask_path)
        self.index: Dict[str, Tuple[Path, Path]] = {}

        # OASIS PNG: images are 'case_*.nii.png', masks are 'seg_*.nii.png'
        # Create dictionaries of all images for O(1) lookups later on
        img_lookup, msk_lookup = {}, {}
        pairs = [
            ("keras_png_slices_train",    "keras_png_slices_seg_train"),
            ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
            ("keras_png_slices_test",     "keras_png_slices_seg_test"),
        ]

        for img_dir, seg_dir in pairs:

            for p in (self.data_dir / img_dir).glob("*.png"):
                stem = p.stem[:-4] if p.stem.endswith(".nii") else p.stem
                if stem.startswith("case_"): img_lookup[stem] = p

            for p in (self.data_dir / seg_dir).glob("*.png"):
                stem = p.stem[:-4] if p.stem.endswith(".nii") else p.stem
                if stem.startswith("seg_"):  msk_lookup[stem] = p

        # Create index for all images; confirm all images in self.ids is present
        for cid in self.ids:
            img_p = img_lookup.get(cid)
            msk_p = msk_lookup.get(cid.replace("case_", "seg_", 1))
            if img_p is None or msk_p is None:
                raise FileNotFoundError(f"Missing pair for id={cid}: img={img_p}, mask={msk_p}")
            self.index[cid] = (img_p, msk_p)

    def _first_existing(self, base: Path, cid: str, exts: List[str]) -> Optional[Path]:
        """Try multiple extensions (and .nii.png) for flat layout convenience."""
        for p in [*(base / f"{cid}{e}" for e in exts), base / f"{cid}.nii.png"]:
            if p.exists(): return p
        return None

    def __len__(self): return len(self.ids)

    def _apply_remap(self, lab: np.ndarray) -> np.ndarray:
        """Convert raw codes (0/85/170/255) to contiguous IDs (0..C-1)."""
        if self.remap is None: return lab
        out = lab.copy()
        for src, dst in self.remap.items(): out[lab == src] = dst
        return out

    def _random_augs(self, img_t: torch.Tensor, lab_t: torch.Tensor):
        """Anatomy-safe light augs: flips and small rotations."""
        if random.random() < 0.5:
            img_t = TF.hflip(img_t); lab_t = TF.hflip(lab_t.unsqueeze(0)).squeeze(0).long()
        if random.random() < 0.5:
            img_t = TF.vflip(img_t); lab_t = TF.vflip(lab_t.unsqueeze(0)).squeeze(0).long()
        if random.random() < 0.2:
            angle = random.uniform(-10, 10)
            img_t = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
            lab_t = TF.rotate(lab_t.unsqueeze(0).float(), angle,
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
        return img_t, lab_t

    def __getitem__(self, idx: int):
        cid = self.ids[idx]
        img_p, msk_p = self.index[cid]

        # Load arrays
        img = _imread_float(str(img_p))      # (H, W) float32
        lab = _imread_label(str(msk_p))      # (H, W) int64

        # Map labels if needed
        lab = self._apply_remap(lab)

        # Per-image z-score normalization
        img = (img - img.mean()) / (img.std() + 1e-8)

        # To tensors; images must be channels-first
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        lab_t = torch.from_numpy(lab).long()        # (H, W)

        if self.augment:
            img_t, lab_t = self._random_augs(img_t, lab_t)

        return img_t, lab_t, cid