# Allow running as a script directly on Unix-like systems.
#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# train.py — Train the Improved 2D U-Net on OASIS PNG slices using OasisSeg2D.
#
# Requires:
#   - dataset.py with class OasisSeg2D(data_dir, id_list_file, num_classes,
#       augment=True, remap_labels_json=None)
#   - modules.py with class ImprovedUNet2D
#   - utils.py with DiceCELoss and dice_per_class
#
# Example:
#   python train.py \
#     --data_dir data \
#     --train_ids splits/train.txt \
#     --val_ids   splits/val.txt \
#     --epochs 30 --batch 4 --device cuda \
#     --deep_supervision --use_se --amp \
#     --num_classes 4 \
#     --save_dir runs/oasis_unet_improved
# -----------------------------------------------------------------------------
"""
train.py — see header comments for usage.
"""

# Read command-line options (e.g., --epochs 30).
import argparse
# Read arguments from defaults json file
import json
# Live estimates for runtime
import time
# File and directory utilities.
import os
# Measure how long training steps take.
import time
# Used in device selection
import contextlib
# Used for maintaining Cuda and DML compatibility
import inspect

# Safer, cleaner path handling.
from pathlib import Path
# Type hints for readability only.
from typing import List, Tuple

# Main ML library we’re using.
import torch
# Functional helpers from PyTorch (not heavily used here).
import torch.nn.functional as F
# Utility to batch and (optionally) shuffle datasets.
from torch.utils.data import DataLoader

# Your dataset: yields (image, mask) given a list of IDs.
from dataset import OasisSeg2D
# Your model: the Improved U-Net.
from modules import ImprovedUNet2D
# Your loss (how wrong) and metric (quality per class).
from utils import DiceCELoss, dice_per_class



# ------------------------- helpers -------------------------
# Build arguments do be parsed at execution
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # ---- Data & splits ----
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    ap.add_argument("--data_dir", type=str, required=False)
    ap.add_argument("--train_ids", type=str, required=False)
    ap.add_argument("--val_ids", type=str, required=False)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--remap_json", type=str, default=None)
    # ---- Model ----
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--use_se", action="store_true")
    ap.add_argument("--deep_supervision", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cpu", "cuda", "dml"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    # ---- Checkpointing ----
    ap.add_argument("--save_dir", type=str, default="runs/oasis_unet_improved")
    return ap

# Allow arguments to be filled using a json file rather than typing at execution
def parse_args_with_config() -> argparse.Namespace:
    # First pass: just to see if --config was provided
    ap = build_parser()
    args, _ = ap.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)

        # Only apply keys that are real args; warn on unknowns
        valid = {a.dest for a in ap._actions if a.dest != "help"}
        unknown = sorted(set(cfg.keys()) - valid)
        if unknown:
            print(f"[config] Warning: ignoring unknown keys: {unknown}")

        # Rebuild parser, set config values as defaults, then parse CLI again
        ap = build_parser()
        ap.set_defaults(**{k: v for k, v in cfg.items() if k in valid})
        return ap.parse_args()
    else:
        return ap.parse_args()

# Count total and trainable parameters for informative printing.
def count_params(model) -> Tuple[int, int]:
    # Total parameters in the model (all numbers).
    total = sum(p.numel() for p in model.parameters())
    # Parameters that will update during training.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Return both counts.
    return total, trainable


# Disable gradient tracking for evaluation; we won't learn here.
@torch.no_grad()
def evaluate(model,
             loader: DataLoader,
             criterion: DiceCELoss,
             device: torch.device,
             num_classes: int) -> Tuple[float, List[float]]:
    # Short doc: returns average loss and average Dice per class.
    """Eval loop: returns (avg_loss, avg_dice_per_class)."""
    # Put model into eval mode (disables training-specific behaviors).
    model.eval()
    # Running sum of losses across validation batches.
    total_loss = 0.0
    # Running sum of per-class Dice across batches.
    dice_sum = torch.zeros(num_classes, device=device)
    # How many batches we evaluated.
    n_batches = 0

    # Iterate through the validation data.
    for images, masks, _ids in loader:
        # Move images to device (CPU/GPU).
        images = images.to(device, non_blocking=True)          # (B, 1, H, W) float32
        # Move masks to device; ensure integer type.
        masks = masks.to(device, non_blocking=True).long()     # (B, H, W)   int64

        # Forward pass: get model outputs (may be a tuple if deep supervision is on).
        outputs = model(images)             # logits or (main, aux2, aux3)
        # If we received multiple heads (deep supervision), combine their losses with weights.
        if isinstance(outputs, tuple):
            # Unpack heads: main and two auxiliaries.
            main, aux2, aux3 = outputs
            # Compute a weighted sum of losses from each head.
            loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
            # Use main head for reporting metrics.
            logits = main
        else:
            # Single-head loss.
            loss = criterion(outputs, masks)
            # Single-head logits for metrics.
            logits = outputs

        # Accumulate loss value for averaging.
        total_loss += float(loss.item())
        # Compute per-class Dice for this batch.
        dpc = dice_per_class(logits, masks, num_classes=num_classes)  # (C,)
        # Add to the running sum.
        dice_sum += dpc.to(device)
        # Increment batch counter.
        n_batches += 1

    # Compute average loss across batches.
    avg_loss = total_loss / max(n_batches, 1)
    # Compute average per-class Dice across batches and convert to a Python list.
    avg_dice_per_class = (dice_sum / max(n_batches, 1)).tolist()
    # Return both.
    return avg_loss, avg_dice_per_class


# Train the model for one full pass through the training set.
def train_one_epoch(model,
                    loader: DataLoader,
                    criterion: DiceCELoss,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler: torch.cuda.amp.GradScaler | None = None,
                    eta_every=10) -> float:

    # Short doc: returns average training loss for this epoch.
    """One training epoch: returns average loss."""
    # Switch model to training mode (enables training-specific behaviors).
    model.train()
    # Running sum of training losses (will average later).
    running = 0.0

    # Iterate through training batches.
    for  b_idx, (images, masks, _ids) in enumerate(loader, 1):
        # start timer
        t0 = time.time()

        # Move images to device.
        images = images.to(device, non_blocking=True)
        # Move masks to device; ensure integer type.
        masks = masks.to(device, non_blocking=True).long()

        # rolling average of batch time
        avg_batch_sec = None

        # Clear old gradients so they don’t accumulate.
        optimizer.zero_grad(set_to_none=True)

        # If gradient scaling is provided (mixed precision), use it for speed/memory benefits on GPU.
        if scaler is not None:
            # Use autocast to run certain ops in lower precision safely.
            with torch.autocast(device_type=device.type,
                                dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16)):
                # Forward pass: model predictions for this batch.
                outputs = model(images)
                # If deep supervision is active, compute a weighted sum of head losses.
                if isinstance(outputs, tuple):
                    main, aux2, aux3 = outputs
                    loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
                else:
                    # Otherwise, a single loss from the sole output.
                    loss = criterion(outputs, masks)
            # Backward pass with scaling to prevent underflow.
            scaler.scale(loss).backward()
            # Optimizer step with scaling.
            scaler.step(optimizer)
            # Update scaler for the next iteration.
            scaler.update()

        # If not using mixed precision, do a standard training step.
        else:
            # Forward pass.
            outputs = model(images)
            # Compute loss (handle deep supervision if present).
            if isinstance(outputs, tuple):
                main, aux2, aux3 = outputs
                loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
            else:
                loss = criterion(outputs, masks)
            # Backward pass: compute gradients.
            loss.backward()
            # Apply parameter updates.
            optimizer.step()

        # Add this batch loss to the running total (for averaging).
        running += float(loss.item())

        # --- rolling ETA (optionally sync for more accurate CUDA timing) ---
        # if device.type == "cuda": torch.cuda.synchronize()  # more accurate, slightly slower
        sec = time.time() - t0
        avg_batch_sec = sec if avg_batch_sec is None else (0.9 * avg_batch_sec + 0.1 * sec)
        if (b_idx % eta_every == 0) or (b_idx == len(loader)):
            remaining = len(loader) - b_idx
            eta = remaining * avg_batch_sec
            mm, ss = divmod(int(eta), 60)
            print(f"[train] {b_idx}/{len(loader)}  loss={loss.item():.4f}  ~ETA {mm}m{ss:02d}s",
                  end="\r", flush=True)

    print()
    # Return average loss over all batches in this epoch.
    return running / max(len(loader), 1)


# --------------------------- main ---------------------------

# Entry point for the script.
def main():
    # Build parser and parse arguments from json file or override at input
    args = parse_args_with_config()

    def choose_device(req: str):
        if req.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            return torch.device(req), True  # (device, amp_ok)
        if req == "dml":
            import torch_directml
            return torch_directml.device(), False  # AMP not supported on DML
        return torch.device("cpu"), False

    # Choose the actual device object safely (fallback to CPU if CUDA not available).
    device, amp_ok = choose_device(args.device)

    # Turn the save directory string into a Path object.
    save_dir = Path(args.save_dir)
    # Create the directory (and parents) if it doesn't exist.
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build the training dataset from your IDs and data folder.
    train_ds = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.train_ids,
        num_classes=args.num_classes,
        augment=True,
        remap_labels_json=args.remap_json,
    )
    # Build the validation dataset (no random augmentations).
    val_ds = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.val_ids,
        num_classes=args.num_classes,
        augment=False,   # no random aug at validation
        remap_labels_json=args.remap_json,
    )

    # Wrap the training dataset in a DataLoader (shuffled).
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          pin_memory=(isinstance(device, torch.device) and device.type=="cuda"),
                          persistent_workers=(args.workers>0))

    # Wrap the validation dataset in a DataLoader (not shuffled).
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # Create the model with your chosen options and move it to the device.
    model = ImprovedUNet2D(
        in_channels=1,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
        dropout=args.dropout,
        use_se=args.use_se,
        deep_supervision=args.deep_supervision
    ).to(device)

    # Count parameters for display.
    total, trainable = count_params(model)
    # Print device and whether mixed precision is on.
    print(f"Device: {device} | AMP: {bool(args.amp and device.type=='cuda')}")
    # Print dataset sizes and chosen batch size.
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | Batch: {args.batch}")
    # Print how many parameters (in millions).
    print(f"Model params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")

    # Build the combined loss function (Cross-Entropy + Dice).
    criterion = DiceCELoss(num_classes=args.num_classes, exclude_bg=False)
    # Build the optimizer that updates model parameters each step.
    def make_adamw(params, lr, weight_decay, device_type: str):
        # CUDA path: prefer fused AdamW if your PyTorch has it
        if device_type == "cuda":
            sig = inspect.signature(torch.optim.AdamW)
            if "fused" in sig.parameters:
                return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
            else:
                # fall back to stock AdamW (PyTorch will pick good CUDA kernels)
                return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            # DML/CPU: avoid multi-tensor foreach path that can hit `lerp`
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, foreach=False)

    optimizer = make_adamw(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, device_type=device.type)

    # Prepare gradient scaler for mixed precision (enabled only if using CUDA + --amp).
    use_amp = bool(args.amp and amp_ok)
    scaler = (torch.amp.GradScaler('cuda') if use_amp else None)
    amp_ctx = (torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext())

    # Keep track of the best (lowest) validation loss seen so far.
    best_val = float("inf")
    # Store training history across epochs for convenience.
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    # Loop over the number of epochs.
    for epoch in range(1, args.epochs + 1):
        # Start timing this epoch.
        t0 = time.time()

        # Train for one full pass over the training set.
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        # Evaluate the model on the validation set (no training here).
        val_loss, val_dice_per_class = evaluate(model, val_loader, criterion, device, args.num_classes)

        # Record training loss for plotting or later analysis.
        history["train_loss"].append(train_loss)
        # Record validation loss likewise.
        history["val_loss"].append(val_loss)
        # Record per-class Dice scores.
        history["val_dice"].append(val_dice_per_class)

        # Average Dice across classes for a single easy-to-read score.
        mean_dice = sum(val_dice_per_class) / max(len(val_dice_per_class), 1)
        # How long the epoch took (seconds).
        dt = time.time() - t0
        # Print a neat progress line with the key numbers.
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={id(train_loss) and train_loss:.4f}  val_loss={id(val_loss) and val_loss:.4f}  "
              f"val_dice(mean)={mean_dice:.4f}  per_class={['%.3f'%d for d in val_dice_per_class]}  "
              f"({dt:.1f}s)")

        # Prepare a checkpoint dictionary holding model and optimizer state.
        ckpt = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "args": vars(args),
        }
        # Always save a "last" checkpoint to allow resuming later.
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))

        # If the validation loss is the best so far, save a "best" checkpoint too.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    # Notify that training is complete.
    print("Training complete.")
    # Print the best validation loss and where checkpoints are stored.
    print(f"Best val loss: {best_val:.4f} | Checkpoints: {save_dir}")


# Run main() if this file is executed directly (not imported).
if __name__ == "__main__":
    main()
