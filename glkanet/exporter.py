"""glkanet/exporter.py — Export 3 bản: train / deploy / onnx."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import torch
import torch.nn as nn


def export_all(
    model:      nn.Module,
    save_dir:   Path,
    input_size: int  = 224,
    yaml_path:  str | Path | None = None,
    opset:      int  = 18,
    verbose:    bool = True,
) -> dict[str, Path]:
    weights_dir = Path(save_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # ── 1. Train weights (chưa reparam) ──────────────────────
    path_train = weights_dir / "best_train.pt"
    torch.save({"state_dict": model.state_dict(), "deployed": False}, path_train)
    if verbose:
        print(f"  [export] train   → {path_train.name}")

    # ── 2. Deploy weights (đã reparam) ───────────────────────
    model_deploy = copy.deepcopy(model)
    model_deploy.eval()
    model_deploy.switch_to_deploy()

    path_deploy = weights_dir / "best_deploy.pt"
    torch.save({"state_dict": model_deploy.state_dict(), "deployed": True}, path_deploy)
    if verbose:
        print(f"  [export] deploy  → {path_deploy.name}")

    # ── 3. ONNX ──────────────────────────────────────────────
    path_onnx = weights_dir / "best_deploy.onnx"

    class _Wrapper(nn.Module):
        """Chỉ trace logits — ONNX không hỗ trợ tuple output tốt."""
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]

    wrapper = _Wrapper(model_deploy)
    wrapper.eval()
    dummy   = torch.zeros(1, 3, input_size, input_size)

    batch          = torch.export.Dim("batch", min=1, max=64)
    dynamic_shapes = {"x": {0: batch}}

    torch.onnx.export(
        wrapper, dummy, str(path_onnx),
        opset_version=max(opset, 18),
        input_names=["images"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
    )
    if verbose:
        print(f"  [export] onnx    → {path_onnx.name}")

    # ── 4. Copy yaml nếu có ──────────────────────────────────
    if yaml_path is not None:
        dst = weights_dir / Path(yaml_path).name
        shutil.copy2(yaml_path, dst)
        if verbose:
            print(f"  [export] yaml    → {dst.name}")

    if verbose:
        _print_sizes(path_train, path_deploy, path_onnx)

    return {"train": path_train, "deploy": path_deploy, "onnx": path_onnx}


def _print_sizes(*paths: Path) -> None:
    print("\n  [export] File sizes:")
    for p in paths:
        if p.exists():
            print(f"           {p.name:<25} {p.stat().st_size/1024/1024:.2f} MB")


def load_checkpoint(
    pt_path:   str | Path,
    yaml_path: str | Path,
    device:    str = "cpu",
):
    """Load model từ .pt — tự detect deployed hay chưa.

    Returns:
        GLKANet ở eval mode
    """
    from glkanet.builder import build_from_yaml

    ckpt  = torch.load(pt_path, map_location=device, weights_only=True)
    model = build_from_yaml(yaml_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt.get("deployed", False):
        model.switch_to_deploy()
    return model.to(device)
