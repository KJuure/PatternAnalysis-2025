import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

mask_dirs = [
    Path("../data/keras_png_slices_seg_train"),
    Path("../data/keras_png_slices_seg_validate"),
    Path("../data/keras_png_slices_seg_test"),
]

checked = 0
for d in mask_dirs:
    for p in d.glob("*.png"):
        arr = np.array(Image.open(p), dtype=np.int64)
        print(p.name, sorted(list(np.unique(arr)))[:10])
        checked += 1
        if checked >= 5:
            raise SystemExit