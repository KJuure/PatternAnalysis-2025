"""
Improved U-Net for 2D medical image segmentation.
Main functions used in the Model

Design highlights
- Residual double-conv blocks: easier optimization, better gradient flow
- InstanceNorm2d + LeakyReLU: stable with small medical batches, keeps small gradients
- Optional Squeeze-and-Excitation (SE) channel attention
- Optional deep supervision: auxiliary logits at mid-decoder during training

I/O
- Input shape : (B, in_channels=1, H, W)
- Output shape: (B, num_classes, H, W); if deep_supervision=True and model.train(),
                returns a tuple (main_logits, aux2_logits, aux3_logits)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Small helpers ----

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """3x3 conv with padding; bias=False because a norm layer follows."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvBlock(nn.Module):
    """
    Residual double 3x3 conv block:

      x ──► Conv3x3 → IN → LReLU → (Dropout) → Conv3x3 → IN → LReLU ──► + ──► out
       └─────────────────────────────(1x1 proj if channels differ)──────┘

    - Residual path lets the block learn corrections over identity → stabler training.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # Project residual if channel count changes (so shapes match for addition)
        self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.res_proj(x)
        x = self.conv1(x); x = self.norm1(x); x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x); x = self.norm2(x); x = self.act(x)
        return x + identity


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (channel attention):
    Global average pool → small bottleneck MLP → sigmoid gates → channel reweighting.
    """
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), kernel_size=1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, output_size=1)  # (B, C, 1, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))               # per-channel gates in [0,1]
        return x * s


class Down(nn.Module):
    """
    Encoder step:
      MaxPool(2) → ConvBlock → (optional) SE attention
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.block(x)
        x = self.se(x)
        return x


class Up(nn.Module):
    """
    Decoder step:
      ConvTranspose2d (×2 upsample) → pad to align with skip → concat(skip, up) → ConvBlock → (optional) SE
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        # We first upsample channels from in_ch to in_ch//2; after concat with skip, channels sum to in_ch again.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-size feature maps: pad upsampled tensor to match skip spatially
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)  # channel-wise concat with encoder skip
        x = self.block(x)
        x = self.se(x)
        return x


class ImprovedUNet2D(nn.Module):
    """
    U-shaped encoder-decoder with residual ConvBlocks, InstanceNorm, LeakyReLU,
    optional SE attention, and optional deep supervision.

    Args:
      in_channels:   input channels (1 for grayscale MRI)
      num_classes:   number of output classes
      base_ch:       base width (doubles each down step)
      dropout:       dropout rate inside ConvBlocks
      use_se:        enable SE blocks
      deep_supervision: if True and model.train(), also returns two aux logits for multi-scale losses
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 base_ch: int = 32,
                 dropout: float = 0.1,
                 use_se: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.inc   = ConvBlock(in_channels, base_ch, dropout=dropout)
        self.down1 = Down(base_ch,   base_ch * 2, dropout=dropout, use_se=use_se)
        self.down2 = Down(base_ch*2, base_ch * 4, dropout=dropout, use_se=use_se)
        self.down3 = Down(base_ch*4, base_ch * 8, dropout=dropout, use_se=use_se)
        self.down4 = Down(base_ch*8, base_ch * 16, dropout=dropout, use_se=use_se)  # bottleneck

        # Decoder
        self.up1 = Up(base_ch*16, base_ch * 8, dropout=dropout, use_se=use_se)
        self.up2 = Up(base_ch*8,  base_ch * 4, dropout=dropout, use_se=use_se)
        self.up3 = Up(base_ch*4,  base_ch * 2, dropout=dropout, use_se=use_se)
        self.up4 = Up(base_ch*2,  base_ch,     dropout=dropout, use_se=use_se)

        # Heads
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        if deep_supervision:
            # Aux logits taken after up1 (C=8*base) and up2 (C=4*base)
            self.ds3 = nn.Conv2d(base_ch * 8, num_classes, kernel_size=1)  # deeper aux
            self.ds2 = nn.Conv2d(base_ch * 4, num_classes, kernel_size=1)  # shallower aux

    def forward(self, x: torch.Tensor):
        # ---- Encoder ----
        x1 = self.inc(x)      # (B, base, H,    W)
        x2 = self.down1(x1)   # (B, 2b,  H/2,  W/2)
        x3 = self.down2(x2)   # (B, 4b,  H/4,  W/4)
        x4 = self.down3(x3)   # (B, 8b,  H/8,  W/8)
        x5 = self.down4(x4)   # (B,16b,  H/16, W/16)

        # ---- Decoder + skip connections ----
        x  = self.up1(x5, x4); aux3 = x  # tap here for deep supervision
        x  = self.up2(x,  x3); aux2 = x
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        logits = self.outc(x)  # (B, num_classes, H, W)

        if self.deep_supervision and self.training:
            # Upsample aux logits to the main resolution so you can apply the same loss
            out3 = F.interpolate(self.ds3(aux3), size=logits.shape[2:], mode="bilinear", align_corners=False)
            out2 = F.interpolate(self.ds2(aux2), size=logits.shape[2:], mode="bilinear", align_corners=False)
            return logits, out2, out3

        return logits
