"""
Losses & metrics for multi-class segmentation (bg/CSF/GM/WM).

- DiceCELoss: combines CrossEntropy and soft Dice
- dice_per_class: per-class Dice from logits + integer masks

Utility Functions used throughout the model
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_labels(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    DML-safe one-hot: works on CPU, CUDA, and DirectML.
    Args:
      target: (B, H, W) int64 class indices in [0, num_classes-1]
      num_classes: number of classes
    Returns:
      (B, C, H, W) float32 one-hot tensor
    """
    if target.dtype != torch.long:
        target = target.long()
    # shape: (1, C, 1, 1)
    classes = torch.arange(num_classes, device=target.device).view(1, num_classes, 1, 1)
    # broadcast compare: (B, 1, H, W) == (1, C, 1, 1) -> (B, C, H, W) boolean
    oh = (target.unsqueeze(1) == classes).to(torch.float32)
    return oh


def soft_dice_per_class(probs: torch.Tensor,
                        target_oh: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    probs:     (B, C, H, W) softmax probabilities
    target_oh: (B, C, H, W) one-hot labels
    returns:   (C,) dice per class, averaged over batch
    """
    dims = (0, 2, 3)  # sum over batch and spatial, keep channel
    intersection = (probs * target_oh).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice


def dice_per_class(logits: torch.Tensor,
                   target: torch.Tensor,
                   num_classes: int,
                   ignore_index: Optional[int] = None) -> torch.Tensor:
    """
    logits: (B, C, H, W) raw scores
    target: (B, H, W)    int64 labels
    returns: (C,) dice per class (NaN for ignored class if applicable)
    """
    if ignore_index is not None:
        # mask ignored pixels to background in both logits and labels
        mask = (target != ignore_index)
        # replace ignored with background (0) so one_hot works; we'll zero them out via mask later
        target = target.clone()
        target[~mask] = 0

    probs = F.softmax(logits, dim=1)                          # (B, C, H, W)
    target_oh = one_hot_labels(target, num_classes)           # (B, C, H, W)

    if ignore_index is not None:
        mask = mask.unsqueeze(1).float()                      # (B,1,H,W)
        probs = probs * mask
        target_oh = target_oh * mask

    return soft_dice_per_class(probs, target_oh)              # (C,)


class DiceCELoss(nn.Module):
    """
    Combined CrossEntropy + soft Dice loss.
    Good default for medical multi-class segmentation.

    Args:
      num_classes: number of classes
      ce_weight:   optional tensor of class weights for CE (shape [C])
      dice_weight: multiplier for Dice term (default 1.0)
      ce_weighting:d multiplier for CE term (default 1.0)
      ignore_index: optional label to ignore in loss/metrics
      exclude_bg:  if True, exclude class 0 from the Dice term
    """
    def __init__(self,
                 num_classes: int,
                 ce_weight: Optional[torch.Tensor] = None,
                 dice_weight: float = 1.0,
                 ce_weighting: float = 1.0,
                 ignore_index: Optional[int] = None,
                 exclude_bg: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # Build CE correctly whether ignore_index is provided or not
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        else:
            self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)

        self.dice_weight = dice_weight
        self.ce_weighting = ce_weighting
        self.ignore_index = ignore_index
        self.exclude_bg = exclude_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE on raw logits + integer target
        ce = self.ce(logits, target)

        # Dice on softmax probabilities vs one-hot labels
        probs = F.softmax(logits, dim=1)                       # (B,C,H,W)
        target_oh = one_hot_labels(target, self.num_classes)   # (B,C,H,W)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)
            probs = probs * mask
            target_oh = target_oh * mask

        dice_c = soft_dice_per_class(probs, target_oh)         # (C,)
        if self.exclude_bg and dice_c.numel() > 1:
            dice = dice_c[1:].mean()
        else:
            dice = dice_c.mean()

        dice_loss = 1.0 - dice
        return self.ce_weighting * ce + self.dice_weight * dice_loss
