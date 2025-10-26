import torch
from modules import ConvBlock

blk = ConvBlock(in_ch=32, out_ch=64, dropout=0.1)
x = torch.randn(2, 32, 128, 128)

# Mimic the forward step-by-step to print shapes
with torch.no_grad():
    identity = blk.res_proj(x)
    print("identity:", identity.shape)               # (2, 64, 128, 128)

    y = blk.conv1(x); print("conv1:", y.shape)       # (2, 64, 128, 128)
    y = blk.norm1(y); print("norm1:", y.shape)
    y = blk.act(y);   print("act1 :", y.shape)
    y = blk.drop(y);  print("drop :", y.shape)

    y = blk.conv2(y); print("conv2:", y.shape)       # (2, 64, 128, 128)
    y = blk.norm2(y); print("norm2:", y.shape)
    y = blk.act(y);   print("act2 :", y.shape)

    out = y + identity
    print("out    :", out.shape)                     # (2, 64, 128, 128)
