"""
Evaluate the Results from the Model
Output easy to read metrics in spreadsheet and figures
"""
#---- Imports ----
# Bring in standard libraries for CLI parsing, JSON, and paths
import argparse, json
from pathlib import Path

# Numerical work + images
import numpy as np
from PIL import Image

# PyTorch core + dataloader
import torch
from torch.utils.data import DataLoader

# Your project modules
from dataset import OasisSeg2D
from modules import ImprovedUNet2D


# ---- Argument parser (config-aware, same style as train.py) ----
def build_parser():
    # Create a new argument parser
    ap = argparse.ArgumentParser()
    # Add an optional JSON config path (lets you avoid long CLI commands)
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config.")

    # You provide: which IDs to run, which weights to load, and where to save outputs
    ap.add_argument("--ids", type=str, required=False, help="ID list (e.g., splits/test.txt or splits/val.txt)")
    ap.add_argument("--weights", type=str, required=False, help="Path to model weights (.pt)")
    ap.add_argument("--out_dir", type=str, required=False, help="Directory to write outputs (masks/overlays/plots/metrics)")

    # Common fields (will default from train_config.json if present)
    ap.add_argument("--data_dir", type=str, required=False)
    ap.add_argument("--num_classes", type=int, default=4)

    # If your GT masks were color-coded (e.g., 0/85/170/255), pass the same remap JSON used in training
    ap.add_argument("--remap_json", type=str, default=None,
                    help="JSON mapping raw mask values -> class IDs 0..C-1 (same as training).")

    # Model structure & runtime options (must match training)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--use_se", action="store_true")
    ap.add_argument("--deep_supervision", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)

    # Inference batch size, DataLoader workers, and device
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "dml"])

    # Pretty outputs for the report (how many overlays to save; optional class names)
    ap.add_argument("--num_overlays", type=int, default=12, help="Save up to N example overlays + error maps")
    ap.add_argument("--class_names", type=str, default=None,
                    help="Comma-separated class names (e.g., 'bg,csf,gm,wm')")
    # Return the filled parser
    return ap


def parse_args_with_config():
    # Build the base parser
    ap = build_parser()
    # Parse once to see if --config was given
    args, _ = ap.parse_known_args()

    # If a JSON config was provided, load it and set those values as defaults
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        # Only keep keys that are real arguments
        valid = {a.dest for a in ap._actions if a.dest != "help"}
        # Rebuild the parser and apply defaults from the config
        ap = build_parser()
        ap.set_defaults(**{k: v for k, v in cfg.items() if k in valid})

    # Parse again so CLI overrides config defaults
    return ap.parse_args()


# ---- Device chooser: CPU / CUDA / DirectML (AMD) ----
def choose_device(req: str):
    # If user asked for CUDA, ensure it’s available, then return a CUDA device
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(req)
    # If user asked for DirectML (AMD on Windows), return a DML device
    if req == "dml":
        import torch_directml
        return torch_directml.device()
    # Otherwise default to CPU
    return torch.device("cpu")


# ---- Simple color palette for overlays (RGBA) ----
PALETTE = {
    # Background transparent
    0: (0, 0, 0, 0),
    # Class 1 = red-ish semi-transparent
    1: (255, 0, 0, 120),
    # Class 2 = green-ish semi-transparent
    2: (0, 200, 0, 120),
    # Class 3 = blue-ish semi-transparent
    3: (0, 120, 255, 120),
}


def colorize_mask(mask_hw: np.ndarray) -> Image.Image:
    # Extract height and width from the integer mask
    h, w = mask_hw.shape
    # Create an empty RGBA image buffer
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # For each class index present, paint pixels with that class’s RGBA color
    for k, c in PALETTE.items():
        if k <= mask_hw.max():
            rgba[mask_hw == k] = c
    # Convert the numpy array to a PIL RGBA image
    return Image.fromarray(rgba, mode="RGBA")


def overlay_pair(img_chw: torch.Tensor, pred_hw: np.ndarray, gt_hw: np.ndarray | None) -> Image.Image:
    # Convert the single-channel float image (C,H,W) in [0,1] to a grayscale PIL RGBA base
    img = (img_chw.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    base = Image.fromarray(img, mode="L").convert("RGBA")
    # Create a color overlay from the prediction
    pred_rgba = colorize_mask(pred_hw)
    pred_comp = Image.alpha_composite(base, pred_rgba)
    # If GT is provided, also create a GT composite; then tile [Input | GT | Pred]
    if gt_hw is not None:
        gt_rgba = colorize_mask(gt_hw)
        gt_comp = Image.alpha_composite(base, gt_rgba)
        canvas = Image.new("RGBA", (base.width * 3, base.height), (0, 0, 0, 0))
        canvas.paste(base, (0, 0))
        canvas.paste(gt_comp, (base.width, 0))
        canvas.paste(pred_comp, (base.width * 2, 0))
        return canvas
    # If no GT, tile [Input | Pred]
    canvas = Image.new("RGBA", (base.width * 2, base.height), (0, 0, 0, 0))
    canvas.paste(base, (0, 0))
    canvas.paste(pred_comp, (base.width, 0))
    return canvas


def error_map(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    # Create a TP/FP/FN visualization (per-pixel) to diagnose errors
    h, w = gt.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # True positive: predicted equals ground truth and not background
    tp = (pred == gt) & (gt != 0)
    # False positive: predicted non-background where GT is background
    fp = (pred != 0) & (gt == 0)
    # False negative: GT non-background where prediction is background
    fn = (gt != 0) & (pred == 0)
    # Color-code the errors and TP pixels
    rgba[tp] = (0, 200, 0, 150)     # green
    rgba[fp] = (0, 120, 255, 150)   # blue
    rgba[fn] = (255, 0, 0, 150)     # red
    # Convert the numpy array to a PIL RGBA image
    return Image.fromarray(rgba, mode="RGBA")


def tile_images(img_paths: list[Path], grid_cols=4, bg=(255, 255, 255, 255)) -> Image.Image | None:
    # Load images listed in img_paths (skip if empty)
    imgs = [Image.open(p).convert("RGBA") for p in img_paths]
    if not imgs:
        return None
    # Assume all the same size; compute grid geometry
    w, h = imgs[0].size
    rows = (len(imgs) + grid_cols - 1) // grid_cols
    # Create a big canvas and paste each image into its grid slot
    canvas = Image.new("RGBA", (w * grid_cols, h * rows), bg)
    for i, im in enumerate(imgs):
        r, c = divmod(i, grid_cols)
        canvas.paste(im, (c * w, r * h))
    # Return the tiled image (good for reports)
    return canvas


# ---- Metrics helpers (NumPy CPU, works on any backend) ----
def update_confusion(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                     pred: np.ndarray, gt: np.ndarray, num_classes: int):
    # For each class, count pixel-wise TP, FP, FN and accumulate them
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        tp[c] += np.logical_and(p, g).sum()
        fp[c] += np.logical_and(p, np.logical_not(g)).sum()
        fn[c] += np.logical_and(np.logical_not(p), g).sum()


def metrics_from_confusion(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    # Use safe denominators to avoid division by zero
    eps0 = 1e-8
    # Dice = 2TP / (2TP + FP + FN)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, eps0)
    # IoU = TP / (TP + FP + FN)
    iou = tp / np.maximum(tp + fp + fn, eps0)
    # Precision = TP / (TP + FP)
    prec = tp / np.maximum(tp + fp, eps0)
    # Recall = TP / (TP + FN)
    rec = tp / np.maximum(tp + fn, eps0)
    # Macro mean (simple average across classes)
    macro = {
        "dice": float(dice.mean()) if dice.size else 0.0,
        "iou": float(iou.mean()) if iou.size else 0.0,
        "precision": float(prec.mean()) if prec.size else 0.0,
        "recall": float(rec.mean()) if rec.size else 0.0,
    }
    # Return per-class arrays and macro dict
    return dice, iou, prec, rec, macro


def save_bar(metric: np.ndarray, names: list[str], title: str, out_png: Path):
    # Use a headless backend so saving figures works without a display
    import matplotlib
    matplotlib.use("Agg")
    # Import pyplot after backend selection
    import matplotlib.pyplot as plt
    # Create a simple bar chart for the metric
    plt.figure()
    xs = np.arange(len(metric))
    plt.bar(xs, metric)
    plt.xticks(xs, names, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    # Ensure parent directory exists, then save the plot
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


# ---- Main prediction/evaluation entry point ----
@torch.no_grad()
def main():
    # Parse config/CLI args (CLI overrides config)
    args = parse_args_with_config()
    # Choose runtime device (CPU / CUDA / DirectML)
    device = choose_device(args.device)

    # If --ids was not given, default to val_ids from the training config
    if args.ids is None:
        if hasattr(args, "val_ids") and args.val_ids:
            args.ids = args.val_ids
            print(f"[predict] Using val_ids from config: {args.ids}")
        else:
            raise AssertionError("--ids must be provided (or present as val_ids in config)")

    if not args.weights or not args.out_dir:
        raise SystemExit("Please provide --weights and --out_dir (either in --config or CLI).")

    # Create the output directory tree (masks/overlays/errors/plots/metrics)
    out_root = Path(args.out_dir)
    out_masks = out_root / "masks"
    out_over = out_root / "overlays"
    out_errs = out_root / "errors"
    out_plots = out_root / "plots"
    out_metrics = out_root / "metrics"
    for d in [out_masks, out_over, out_errs, out_plots, out_metrics]:
        d.mkdir(parents=True, exist_ok=True)

    # If provided, make remap_json path absolute (relative to data_dir)
    remap_path = None
    if args.remap_json:
        p = Path(args.remap_json)
        remap_path = str(p)

    # Parse class names if provided; otherwise default to class_0..class_C-1
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]
        if len(class_names) != args.num_classes:
            print(f"[predict] Warning: class_names count != num_classes; using defaults class_0..class_{args.num_classes-1}.")
            class_names = [f"class_{i}" for i in range(args.num_classes)]
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    # Build dataset and loader (augment=False for prediction/eval)
    ds = OasisSeg2D(args.data_dir, args.ids, args.num_classes, augment=False, remap_labels_json=remap_path)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(isinstance(device, torch.device) and device.type == "cuda"),
    )

    # Construct the model to match training, move to device, and load weights
    model = ImprovedUNet2D(
        in_channels=1,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
        dropout=args.dropout,
        use_se=args.use_se,
        deep_supervision=args.deep_supervision,
    ).to(device)

    # Load checkpoint (support both raw state dicts and dicts with 'state_dict')
    # Load the checkpoint on CPU for maximum compatibility (works for CUDA/DML/CPU)
    # Always load checkpoint tensors onto CPU for compatibility
    ckpt = torch.load(args.weights, map_location="cpu")

    # 1) Pick the actual state_dict regardless of how it was saved
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        else:
            # If ckpt already looks like a state_dict (param-name -> tensor)
            k0 = next(iter(ckpt))
            state_dict = ckpt if isinstance(ckpt[k0], torch.Tensor) else None
    else:
        state_dict = None

    if state_dict is None:
        raise RuntimeError("Could not locate a state_dict in the checkpoint.")

    # 2) Strip common prefixes so names match your ImprovedUNet2D modules
    state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}

    # 3) Load into the model already moved to `device`
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 4) Optional diagnostics (won’t stop execution)
    if missing:
        print("[predict] Warning: missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[predict] Warning: unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    # Put the model in evaluation mode (disables dropout, etc.)
    model.eval()

    # Prepare per-class TP/FP/FN accumulators
    C = args.num_classes
    tp = np.zeros(C, dtype=np.int64)
    fp = np.zeros(C, dtype=np.int64)
    fn = np.zeros(C, dtype=np.int64)

    # Track overlay images to build a tiled grid later
    saved_ov = 0
    overlay_paths = []

    # Count how many masks were written
    total_saved_masks = 0

    # Iterate over batches from the loader
    for batch in loader:
        # Your dataset returns (image, mask, id); if it changes, adapt here
        if isinstance(batch, (list, tuple)):
            images, masks, ids = batch[0], batch[1], batch[2]
        else:
            # Fallback for dict-style datasets (not used in your current setup)
            images, ids = batch["image"], batch["id"]
            masks = None

        # Move images to device (masks not needed for writing predictions, but needed for metrics)
        images = images.to(device)

        # Forward pass to get per-class logits
        logits = model(images)

        # If deep supervision returns multiple heads, use the main (highest-res) head at index 0
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert logits to predicted class IDs via argmax over channels
        preds = torch.argmax(logits, dim=1)  # shape (B, H, W), ints in [0..C-1]

        # For each item in the batch, save mask and (optionally) visuals/metrics
        B = images.size(0)
        for i in range(B):
            # Grab a unique ID for consistent filenames
            cid = ids[i]

            # Convert prediction tensor to a numpy array
            pred_hw = preds[i].cpu().numpy().astype(np.uint8)  # class IDs 0..C-1
            mask_path = out_masks / f"{cid}_pred.png"
            Image.fromarray(pred_hw, mode="L").save(mask_path)

            total_saved_masks += 1

            # If ground truth exists (val split), update metrics and save overlays/error maps
            if masks is not None:
                gt_hw = masks[i].cpu().numpy().astype(np.int64)
                update_confusion(tp, fp, fn, pred_hw, gt_hw, C)

                # Save a limited number of report visuals (controlled by --num_overlays)
                if saved_ov < args.num_overlays:
                    # Build an input|GT|Pred side-by-side overlay
                    ov = overlay_pair(images[i].cpu(), pred_hw, gt_hw)
                    ov_path = out_over / f"{cid}_overlay.png"
                    ov.save(ov_path)
                    overlay_paths.append(ov_path)

                    # Build a TP/FP/FN error map blended over the input
                    base = (images[i].cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                    base_rgba = Image.fromarray(base, mode="L").convert("RGBA")
                    em = error_map(gt_hw, pred_hw)
                    err_path = out_errs / f"{cid}_errors.png"
                    Image.alpha_composite(base_rgba, em).save(err_path)

                    # Update the overlay counter
                    saved_ov += 1

    # After processing all batches, compute and save metrics if we had GT masks
    if tp.sum() + fp.sum() + fn.sum() > 0:
        # Compute per-class arrays and macro means
        dice, iou, prec, rec, macro = metrics_from_confusion(tp, fp, fn)

        # Print a human-readable table to the console
        print("\n=== Metrics (pixel-wise) ===")
        print("Class\t\tDice\tIoU\tPrec\tRecall")
        for i, name in enumerate(class_names):
            print(f"{name:10s}\t{dice[i]:.4f}\t{iou[i]:.4f}\t{prec[i]:.4f}\t{rec[i]:.4f}")
        print(f"Macro mean\t{macro['dice']:.4f}\t{macro['iou']:.4f}\t{macro['precision']:.4f}\t{macro['recall']:.4f}")

        # Save a CSV with per-class metrics and macro means (great for your report appendix)
        import csv
        with open(out_metrics / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "dice", "iou", "precision", "recall", "tp", "fp", "fn"])
            for i, name in enumerate(class_names):
                w.writerow([name, f"{dice[i]:.6f}", f"{iou[i]:.6f}", f"{prec[i]:.6f}", f"{rec[i]:.6f}",
                            int(tp[i]), int(fp[i]), int(fn[i])])
            w.writerow(["macro_mean", f"{macro['dice']:.6f}", f"{macro['iou']:.6f}",
                        f"{macro['precision']:.6f}", f"{macro['recall']:.6f}", "", "", ""])

        # Save per-class bar charts (Dice & IoU) to paste into the report
        save_bar(dice, class_names, "Per-class Dice", out_plots / "dice_bar.png")
        save_bar(iou,  class_names, "Per-class IoU",  out_plots / "iou_bar.png")

        # Build a tiled grid of overlay examples
        grid = tile_images(overlay_paths, grid_cols=4)
        if grid is not None:
            grid.save(out_plots / "overlay_grid.png")

    # Final console summary of where files went
    print(f"[predict] Saved {total_saved_masks} masks to: {out_masks}")
    if saved_ov:
        print(f"[predict] Overlays: {saved_ov} saved to {out_over}")
        print(f"[predict] Error maps saved to {out_errs}")
        print(f"[predict] Plots/metrics at {out_root}")


# ---- Script entry point ----
if __name__ == "__main__":
    # Run the main function when this file is executed directly
    main()
