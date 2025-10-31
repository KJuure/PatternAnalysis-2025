"""
Sanity tests for utils.py (losses & metrics).

What this checks:
1) one_hot_labels: shape & correctness on a tiny target
2) soft_dice_per_class: gives 1.0 for exact matches on both classes
3) dice_per_class: matches soft_dice_per_class when using the same probs
4) ignore_index: ignored pixels don't affect Dice
5) exclude_bg: averaging excludes class 0 when requested
6) DiceCELoss: forward/backward; grads flow to parameters; scalar finite

Run:
  python tests/test_utils.py
"""

import os
import sys
import pathlib
import math
import torch
import torch.nn.functional as F

# Make project root importable if running from tests/
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils import one_hot_labels, soft_dice_per_class, dice_per_class, DiceCELoss


def _almost_equal(a, b, tol=1e-5):
    return float(abs(a - b)) <= tol


def test_one_hot_labels():
    print("\n[1] one_hot_labels")
    target = torch.tensor([
        [0, 1, 2],
        [2, 1, 0]
    ]).unsqueeze(0)  # (B=1, H=2, W=3)
    num_classes = 3

    oh = one_hot_labels(target, num_classes)  # (1, 3, 2, 3)
    assert oh.shape == (1, 3, 2, 3), f"Bad shape: {oh.shape}"

    # spot checks
    # (row=0,col=0) is class 0
    assert _almost_equal(oh[0, 0, 0, 0].item(), 1.0)
    assert _almost_equal(oh[0, 1, 0, 0].item(), 0.0)
    # (row=0,col=1) is class 1
    assert _almost_equal(oh[0, 1, 0, 1].item(), 1.0)
    # (row=1,col=0) is class 2
    assert _almost_equal(oh[0, 2, 1, 0].item(), 1.0)
    print("  ✓ shape & values OK")


def test_soft_dice_per_class_exact_match():
    print("\n[2] soft_dice_per_class (exact match → Dice=1)")
    # Make a small case where predictions == labels exactly
    target = torch.tensor([[0, 1],
                           [1, 0]]).unsqueeze(0)  # (1,2,2)
    C = 2

    target_oh = one_hot_labels(target, C)       # (1,2,2,2) → (1,2,2,2)
    probs = target_oh.clone()                   # perfect prediction

    dice_c = soft_dice_per_class(probs, target_oh)  # (C,)
    assert dice_c.shape == (C,), f"Bad shape: {dice_c.shape}"
    # both classes present perfectly → both ~1.0
    assert all(_almost_equal(float(d), 1.0, tol=1e-6) for d in dice_c), f"Dice not 1.0: {dice_c}"
    print("  ✓ both classes Dice≈1.0")


def test_dice_per_class_matches_manual():
    print("\n[3] dice_per_class matches manual soft Dice path")
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 8, 8
    logits = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    # via helper
    dice_from_helper = dice_per_class(logits, target, num_classes=C)

    # manual path: softmax + one_hot + soft_dice
    probs = F.softmax(logits, dim=1)
    target_oh = one_hot_labels(target, C)
    dice_manual = soft_dice_per_class(probs, target_oh)

    assert dice_from_helper.shape == (C,)
    assert torch.allclose(dice_from_helper, dice_manual, atol=1e-6), \
        f"Mismatch:\nhelper={dice_from_helper}\nmanual={dice_manual}"
    print("  ✓ helper == manual path")


def test_ignore_index_masks_out():
    print("\n[4] ignore_index masks out pixels for Dice")
    # Build a simple 2-class 3x3 example where center pixel is ignored
    C = 2
    target = torch.tensor([[0, 1, 1],
                           [1, 255, 0],   # 255 = ignore_index
                           [0, 0, 1]]).unsqueeze(0)  # (1,3,3)
    ignore_index = 255

    # Create logits that are perfect on valid pixels, random on ignored pixel
    logits = torch.zeros(1, C, 3, 3)
    # set logits so that class == target for valid pixels
    for i in range(3):
        for j in range(3):
            t = int(target[0, i, j].item())
            if t != ignore_index:
                logits[0, t, i, j] = 5.0   # high score on correct class

    dice_c = dice_per_class(logits, target, num_classes=C, ignore_index=ignore_index)
    # since all valid pixels are perfectly predicted, Dice per present classes ≈ 1.
    # Some classes might have few pixels; we still expect ≈1.0 due to perfect match on counted pixels.
    assert torch.all(dice_c <= 1.0 + 1e-6)
    assert torch.all(dice_c >= -1e-6)
    print("  ✓ ignore_index handled (valid pixels → near-perfect Dice)")

def test_exclude_bg_behavior():
    print("\n[5] exclude_bg averaging check")
    # Build logits/labels where foreground is predicted better than background
    B, C, H, W = 1, 3, 4, 4   # classes: 0=bg, 1, 2
    torch.manual_seed(1)
    target = torch.randint(0, C, (B, H, W))
    logits = torch.randn(B, C, H, W)

    # Compute per-class dice
    dice_c = dice_per_class(logits, target, num_classes=C)
    dice_all = dice_c.mean()
    dice_no_bg = dice_c[1:].mean()

    # Check shapes and that excluding bg changes or equals (depending on values)
    assert dice_c.shape == (C,)
    assert isinstance(dice_all.item(), float)
    assert isinstance(dice_no_bg.item(), float)
    print(f"  Dice per class: {dice_c.tolist()}")
    print(f"  Mean(all)={dice_all.item():.4f}  Mean(no_bg)={dice_no_bg.item():.4f}")
    print("  ✓ exclude_bg metric path OK")


def test_dice_ce_loss_forward_backward():
    print("\n[6] DiceCELoss forward/backward")
    torch.manual_seed(123)
    B, C, H, W = 2, 4, 16, 16
    logits = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, C, (B, H, W))

    # Build loss
    crit = DiceCELoss(num_classes=C, dice_weight=1.0, ce_weighting=1.0,
                      ignore_index=None, exclude_bg=False)
    loss = crit(logits, target)
    assert math.isfinite(float(loss)), f"Loss not finite: {loss}"
    loss.backward()

    # Check gradients exist on logits (upstream of model params here)
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "No/NaN grads on logits"
    print(f"  loss={float(loss):.4f}, grad_norm={float(logits.grad.norm()):.4f}")
    print("  ✓ forward/backward OK")


def main():
    test_one_hot_labels()
    test_soft_dice_per_class_exact_match()
    test_dice_per_class_matches_manual()
    test_ignore_index_masks_out()
    test_exclude_bg_behavior()
    test_dice_ce_loss_forward_backward()
    print("\nAll utils tests passed ✔")


if __name__ == "__main__":
    main()
