"""
Đo kích thước model: tổng params, FLOPs (MACs), phân tích từng layer.
Không phụ thuộc vào UI — chỉ nhận model / state_dict, trả về dict.
"""
from __future__ import annotations
from typing import Any

from .deps import HAS_TORCH, HAS_THOP
from ..utils import fmt_flops


def count_params_from_state_dict(sd: dict) -> dict:
    """
    Đếm params, tính dung lượng bộ nhớ, phân tích dtype từ state_dict.
    Không cần forward pass.
    """
    if not HAS_TORCH:
        return {}
    import torch

    total_params = sum(
        v.numel()
        for v in sd.values()
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
    )
    total_bytes = sum(
        v.numel() * v.element_size()
        for v in sd.values()
        if isinstance(v, torch.Tensor)
    )

    dtype_counts: dict[str, int] = {}
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            k = str(v.dtype)
            dtype_counts[k] = dtype_counts.get(k, 0) + 1

    layers = [
        {
            "name":   name,
            "shape":  list(tensor.shape),
            "params": tensor.numel(),
            "dtype":  str(tensor.dtype),
        }
        for name, tensor in sd.items()
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point
    ]

    return {
        "total_params":  total_params,
        "model_size_mb": total_bytes / 1e6,
        "dtypes":        dtype_counts,
        "layers":        layers,
    }


def compute_flops(model: Any, input_tensor: Any) -> dict:
    """
    Tính FLOPs (MACs) dùng thop.

    - Params đếm trực tiếp từ model.parameters() — KHÔNG qua thop
      và KHÔNG wrap thêm để tránh double-count.
    - Chỉ wrap output nếu model trả về tuple/list (thop cần tensor đơn).
    """
    if not HAS_THOP:
        return {"flops_err": "pip install thop để tính FLOPs"}

    import torch
    import torch.nn as nn

    try:
        # ── Params chính xác từ model gốc ─────────────────────────────────
        total_params = sum(p.numel() for p in model.parameters())

        # ── Wrap output nếu cần — chỉ để thop hoạt động, không đếm params ─
        class _OutputWrap(nn.Module):
            """Chỉ chuẩn hóa output, không thêm params."""
            def __init__(self, m: nn.Module):
                super().__init__()
                self.m = m
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                o = self.m(x)
                return o[0] if isinstance(o, (tuple, list)) else o

        from thop import profile as thop_profile
        # verbose=False để tránh spam log
        flops, _ = thop_profile(
            _OutputWrap(model),
            inputs=(input_tensor,),
            verbose=False,
        )

        return {
            "flops":       int(flops),
            "flops_str":   fmt_flops(flops),
            "params_thop": total_params,    # params từ model gốc, không phải thop
        }

    except Exception as ex:
        return {"flops_err": str(ex)}


def infer_input_size(sd: dict, fallback: int = 224) -> int:
    """
    Đoán input spatial size từ state_dict.
    Ưu tiên: tìm stem conv weight 4D đầu tiên → dùng fallback nếu không đoán được.
    Với Simple_GLKA input_size=224 nên fallback mặc định đúng rồi.
    """
    if not HAS_TORCH:
        return fallback
    import torch

    # Không có cách reliable đoán spatial size từ weight shapes
    # → trả fallback 224 (đúng với Simple_GLKA)
    return fallback