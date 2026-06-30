"""export_onnx.py — Export lại ONNX từ checkpoint đã train, không cần train lại."""

import torch
import torch.nn as nn
from pathlib import Path

from glkanet.exporter import load_checkpoint

# ── Sửa các đường dẫn này cho đúng với máy bạn ──────────────
PT_PATH    = "runs/exp3/weights/best_deploy.pt"   # đường dẫn thật tới checkpoint
YAML_PATH  = "glkanet/configs/shuffle_glka.yaml"  # config kiến trúc — dùng đúng file bạn dùng để build model lúc train (không phải train.yaml, train.yaml là config train chứ không phải config kiến trúc)
SAVE_PATH  = "runs/exp3/weights/best_deploy.onnx"
INPUT_SIZE = 224
OPSET      = 18
# ─────────────────────────────────────────────────────────────


class _Wrapper(nn.Module):
    """Chỉ trace logits — ONNX không hỗ trợ tuple output tốt."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)[0]


def main():
    # Load model (load_checkpoint tự detect deployed hay chưa,
    # và tự gọi switch_to_deploy() nếu cần)
    model = load_checkpoint(PT_PATH, YAML_PATH, device="cpu")
    model.eval()

    wrapper = _Wrapper(model).cpu()
    wrapper.eval()

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)  # CPU, khớp với wrapper

    dynamic_axes = {
        "images": {0: "batch"},
        "logits": {0: "batch"},
    }

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(save_path),
        opset_version=max(OPSET, 18),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[export] onnx → {save_path}")
    print(f"[export] size: {save_path.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()