from pathlib import Path

"""
Generate train.txt, val.txt and test.txt files for given data
Used in locating each image
"""
root = Path("data")
out  = Path("splits"); out.mkdir(exist_ok=True)

def write_ids(img_dir, out_file):
    """
    Takes the image directory and generates a corresponding ID text file at the given output

    :param img_dir:
    :param out_file:
    """
    ids = []
    for p in (root / img_dir).glob("*.png"):
        stem = p.stem
        # turn "case_001_slice_0.nii" -> "case_001_slice_0"
        if stem.endswith(".nii"):
            stem = stem[:-4]
        if stem.startswith("case_"):
            ids.append(stem)
    ids = sorted(set(ids))
    Path(out_file).write_text("\n".join(ids))
    print(f"Wrote {len(ids)} ids -> {out_file}")

write_ids("keras_png_slices_train",    "splits/train.txt")
write_ids("keras_png_slices_validate", "splits/val.txt")
write_ids("keras_png_slices_test",     "splits/test.txt")