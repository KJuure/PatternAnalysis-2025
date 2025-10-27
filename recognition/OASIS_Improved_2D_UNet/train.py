"""
train.py — Train the Improved 2D U-Net on OASIS PNG slices using OasisSeg2D.

Requires:
  - dataset.py with class OasisSeg2D(data_dir, id_list_file, num_classes, augment=True, remap_labels_json=None)
  - modules.py with class ImprovedUNet2D
  - utils.py with DiceCELoss and dice_per_class

Example:
  python train.py \
    --data_dir data \
    --train_ids splits/train.txt \
    --val_ids   splits/val.txt \
    --epochs 30 --batch 4 --device cuda \
    --deep_supervision --use_se --amp \
    --num_classes 4 \
    --save_dir runs/oasis_unet_improved
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import OasisSeg2D
from modules import ImprovedUNet2D
from utils import DiceCELoss, dice_per_class


# ------------------------- helpers -------------------------

def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model,
             loader: DataLoader,
             criterion: DiceCELoss,
             device: torch.device,
             num_classes: int) -> Tuple[float, List[float]]:
    """Eval loop: returns (avg_loss, avg_dice_per_class)."""
    model.eval()
    total_loss = 0.0
    dice_sum = torch.zeros(num_classes, device=device)
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)          # (B, 1, H, W) float32
        masks = masks.to(device).long()     # (B, H, W)   int64

        outputs = model(images)             # logits or (main, aux2, aux3)
        if isinstance(outputs, tuple):
            main, aux2, aux3 = outputs
            loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
            logits = main
        else:
            loss = criterion(outputs, masks)
            logits = outputs

        total_loss += float(loss.item())
        dpc = dice_per_class(logits, masks, num_classes=num_classes)  # (C,)
        dice_sum += dpc.to(device)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice_per_class = (dice_sum / max(n_batches, 1)).tolist()
    return avg_loss, avg_dice_per_class


def train_one_epoch(model,
                    loader: DataLoader,
                    criterion: DiceCELoss,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler: torch.cuda.amp.GradScaler | None = None) -> float:
    """One training epoch: returns average loss."""
    model.train()
    running = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision branch
            with torch.autocast(device_type=device.type,
                                dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16)):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main, aux2, aux3 = outputs
                    loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
                else:
                    loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                main, aux2, aux3 = outputs
                loss = criterion(main, masks) + 0.4 * criterion(aux2, masks) + 0.2 * criterion(aux3, masks)
            else:
                loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running += float(loss.item())

    return running / max(len(loader), 1)


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data & splits
    ap.add_argument("--data_dir", type=str, required=True,
                    help="OASIS root containing keras_png_slices_* and keras_png_slices_seg_* folders.")
    ap.add_argument("--train_ids", type=str, required=True,
                    help="Path to train id list (lines like 'case_001_slice_0').")
    ap.add_argument("--val_ids", type=str, required=True,
                    help="Path to val id list (lines like 'case_001_slice_0').")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--remap_json", type=str, default=None,
                    help="Optional JSON mapping (e.g., {'0':0,'85':1,'170':2,'255':3}).")

    # Model
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--use_se", action="store_true", help="Enable Squeeze-Excite in Down/Up blocks.")
    ap.add_argument("--deep_supervision", action="store_true", help="Use deep supervision (aux losses).")
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for stability.")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on CUDA).")

    # Checkpointing
    ap.add_argument("--save_dir", type=str, default="runs/oasis_unet_improved",
                    help="Directory to save last.pt and best.pt")

    args = ap.parse_args()

    # Device & save dir
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    train_ds = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.train_ids,
        num_classes=args.num_classes,
        augment=True,
        remap_labels_json=args.remap_json,
    )
    val_ds = OasisSeg2D(
        data_dir=args.data_dir,
        id_list_file=args.val_ids,
        num_classes=args.num_classes,
        augment=False,   # no random aug at validation
        remap_labels_json=args.remap_json,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # Model, loss, optimizer, scaler
    model = ImprovedUNet2D(
        in_channels=1,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
        dropout=args.dropout,
        use_se=args.use_se,
        deep_supervision=args.deep_supervision
    ).to(device)

    total, trainable = count_params(model)
    print(f"Device: {device} | AMP: {bool(args.amp and device.type=='cuda')}")
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | Batch: {args.batch}")
    print(f"Model params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")

    criterion = DiceCELoss(num_classes=args.num_classes, exclude_bg=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Train
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice_per_class = evaluate(model, val_loader, criterion, device, args.num_classes)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice_per_class)

        mean_dice = sum(val_dice_per_class) / max(len(val_dice_per_class), 1)
        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_dice(mean)={mean_dice:.4f}  per_class={['%.3f'%d for d in val_dice_per_class]}  "
              f"({dt:.1f}s)")

        # Save last checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))

        # Save best by val loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    print("Training complete.")
    print(f"Best val loss: {best_val:.4f} | Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
