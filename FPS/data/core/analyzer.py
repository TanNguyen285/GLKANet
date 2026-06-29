"""
Điều phối (orchestrate) tất cả loaders.
Đây là entry point duy nhất mà UI/worker cần gọi.

Sơ đồ quyết định:
  .onnx              → loader_onnx
  .pt/.pth (glka)    → loader_glka  (detect qua is_glka_checkpoint)
  .pt/.pth (unknown) → báo lỗi rõ
"""
from __future__ import annotations
import os
import time

from .loader_onnx import load_onnx
from .loader_glka import load_glka, is_glka_checkpoint
from .device      import best_torch_device


def analyze(
    weights_path: str,
    device_str:   str  | None = None,
    script_path:  str  | None = None,   # giữ param để worker không cần sửa
    warmup:       int  | None = None,
    runs:         int  | None = None,
) -> dict:
    """
    Hàm duy nhất UI cần gọi.

    Parameters
    ----------
    weights_path : đường dẫn tới .pt / .pth / .onnx
    device_str   : 'cpu' | 'cuda' | 'mps' — None → tự chọn tốt nhất
    script_path  : không dùng nữa, giữ lại để tương thích với worker
    warmup       : None → default theo device (cuda:200 / cpu:50)
    runs         : None → default theo device (cuda:1000 / cpu:200)
    """
    if device_str is None:
        device_str = best_torch_device()

    ext    = os.path.splitext(weights_path)[1].lower()
    result: dict = {}

    if ext == ".onnx":
        result = load_onnx(
            weights_path,
            device_str=device_str,
            warmup=warmup,
            runs=runs,
        )

    elif ext in (".pt", ".pth"):
        if is_glka_checkpoint(weights_path):
            result = load_glka(
                weights_path,
                device_str=device_str,
                warmup=warmup,
                runs=runs,
            )
        else:
            raise ValueError(
                f"[analyzer] File '{os.path.basename(weights_path)}' không phải "
                f"GLKANet checkpoint.\n"
                f"  GLKANet checkpoint cần có keys: 'state_dict' và 'deployed'.\n"
                f"  Dùng glkanet/exporter.py để export đúng format."
            )

    else:
        raise ValueError(
            f"Định dạng không hỗ trợ: {ext!r}  "
            f"(chỉ nhận .pt / .pth / .onnx)"
        )

    result["path"]        = weights_path
    result["device_used"] = device_str
    result["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result