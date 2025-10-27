"""
Sanity tests for modules.py (ImprovedUNet2D).

What this checks:
1) Forward pass (train mode, deep supervision on/off) -> shapes are correct
2) Forward pass (eval mode) -> returns only main logits
3) Odd H,W inputs -> Up blocks pad correctly so shapes still match
4) Backward pass -> grads flow to parameters
5) Parameter count -> printed for visibility

Run:
  python tests/test_modules.py                        # defaults (cpu, ds off)
  python tests/test_modules.py --deep_supervision     # enable deep supervision
  python tests/test_modules.py --size 255 257         # odd H,W smoke
  python tests/test_modules.py --device cuda          # if you have a GPU
"""

import argparse
import pathlib
import sys
import torch

# If this file lives in tests/, make project root importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from modules import ImprovedUNet2D


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def assert_shape(tensor, shape, msg=""):
    got = tuple(tensor.shape)
    assert got == shape, f"{msg} expected shape {shape}, got {got}"


def test_forward_backward(device: str, in_ch: int, num_classes: int,
                          base_ch: int, H: int, W: int, deep_supervision: bool):
    print("\n== Case ==")
    print(f"device={device} in_ch={in_ch} num_classes={num_classes} base_ch={base_ch} size=({H},{W}) deep_supervision={deep_supervision}")

    model = ImprovedUNet2D(
        in_channels=in_ch,
        num_classes=num_classes,
        base_ch=base_ch,
        dropout=0.1,
        use_se=True,
        deep_supervision=deep_supervision
    ).to(device)

    total, trainable = count_params(model)
    print(f"params: total={total/1e6:.3f}M trainable={trainable/1e6:.3f}M")

    x = torch.randn(2, in_ch, H, W, device=device)  # B=2

    # ---- Train mode (deep supervision: expect tuple; else tensor) ----
    model.train()
    out = model(x)
    if deep_supervision:
        assert isinstance(out, tuple) and len(out) == 3, "With deep_supervision=True (train), model should return (main, aux2, aux3)"
        main, aux2, aux3 = out
        print("train outputs:", tuple(main.shape), tuple(aux2.shape), tuple(aux3.shape))
        # All logits must be (B, C, H, W)
        assert_shape(main, (2, num_classes, H, W), "main logits")
        assert_shape(aux2, (2, num_classes, H, W), "aux2 logits")
        assert_shape(aux3, (2, num_classes, H, W), "aux3 logits")
        # Backward (simple scalar loss)
        loss = main.mean() + 0.4 * aux2.mean() + 0.2 * aux3.mean()
    else:
        assert not isinstance(out, tuple), "With deep_supervision=False, model should return a tensor in train mode"
        print("train output:", tuple(out.shape))
        assert_shape(out, (2, num_classes, H, W), "train logits")
        loss = out.mean()

    # Backprop smoke
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # At least some parameters must have gradients
    any_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert any_grad, "No gradients found after backward()"
    opt.step()
    print("backward: ✓ (grads flowed)")

    # ---- Eval mode (should return ONLY main logits tensor) ----
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert not isinstance(out_eval, tuple), "In eval mode, model must return a single logits tensor (no aux outputs)"
    print("eval output:", tuple(out_eval.shape))
    assert_shape(out_eval, (2, num_classes, H, W), "eval logits")

    print("✓ case passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--in_ch", type=int, default=1)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H","W"))
    ap.add_argument("--deep_supervision", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    H, W = args.size

    # 1) Regular even size
    test_forward_backward(args.device, args.in_ch, args.num_classes, args.base_ch, H, W, args.deep_supervision)

    # 2) Odd size smoke (pads inside Up to match skips)
    #    Use a small odd size so memory stays low.
    oddH, oddW = (H if H % 2 else H - 1), (W if W % 2 else W - 1)
    if oddH < 32: oddH = 255
    if oddW < 32: oddW = 257
    test_forward_backward(args.device, args.in_ch, args.num_classes, args.base_ch, oddH, oddW, args.deep_supervision)

    print("\nAll tests passed ✔")


if __name__ == "__main__":
    main()