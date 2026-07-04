"""export_onnx.py — Export lại ONNX từ checkpoint đã train, không cần train lại.

Đồng bộ đầy đủ với glkanet/exporter.py:
  - _Wrapper an toàn với cả output tuple/list lẫn tensor thẳng
  - onnxsim.simplify() để rút gọn node graph (giảm latency thực tế)
  - validate logits PyTorch deploy vs ONNX runtime sau simplify
  - dynamic batch xuyên suốt (không ép cứng batch=1 khi simplify)
"""

import torch
import torch.nn as nn
from pathlib import Path

from glkanet.exporter import load_checkpoint

# ── Sửa các đường dẫn này cho đúng với máy bạn ──────────────
PT_PATH    = "runs/exp2/weights/best_train.pt"   # đường dẫn thật tới checkpoint
YAML_PATH  = "glkanet/configs/shuffle_glkav2.yaml"  # config kiến trúc — dùng đúng file bạn dùng để build model lúc train (không phải train.yaml, train.yaml là config train chứ không phải config kiến trúc)
SAVE_PATH  = "runs/exp2/weights/best_deploy.onnx"
INPUT_SIZE = 224
OPSET      = 18
VALIDATE   = True     # so sánh logits PyTorch vs ONNX sau export
N_SAMPLES  = 4         # số ảnh dummy dùng để validate (khớp batch dynamic)
ATOL       = 1e-3
RTOL       = 1e-3
# ─────────────────────────────────────────────────────────────


class _Wrapper(nn.Module):
    """Chỉ trace logits — ONNX không hỗ trợ tuple output tốt.
    Tự nhận diện output là tuple/list hay tensor thẳng, không cần sửa khi đổi model khác."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def _get_logits(out) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _validate_onnx(onnx_path: Path, model, input_size: int,
                    atol: float, rtol: float, n_samples: int) -> bool:
    """So sánh output PyTorch deploy vs ONNX runtime. Skip êm nếu chưa cài onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [validate] onnxruntime chưa cài — bỏ qua bước validate ONNX "
              "(pip install onnxruntime --break-system-packages để bật).")
        return True

    dummy = torch.randn(n_samples, 3, input_size, input_size)
    out_torch = _get_logits(model(dummy)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = sess.run(None, {"images": dummy.numpy()})[0]

    import numpy as np
    max_abs_diff = float(np.abs(out_torch - out_onnx).max())
    ok = np.allclose(out_torch, out_onnx, atol=atol, rtol=rtol)

    status = "OK" if ok else "MISMATCH"
    print(f"  [validate] deploy vs onnx logits: max_abs_diff={max_abs_diff:.6f} -> {status}")
    if not ok:
        print("  [validate] CẢNH BÁO: ONNX export lệch so với PyTorch deploy model!")
    return ok


def main():
    # Load model (load_checkpoint tự detect deployed hay chưa,
    # và tự gọi switch_to_deploy() nếu cần — đúng thứ tự trước load_state_dict)
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

    # ── Simplify: rút gọn node graph (Shape/Gather/Cast/Unsqueeze thừa
    #     từ channel_shuffle/reshape), giữ nguyên dynamic batch — KHÔNG
    #     dùng overwrite_input_shapes vì sẽ ép cứng batch=1, mâu thuẫn
    #     với dynamic_axes vừa export (gây lỗi validate batch>1). ──────
    try:
        import onnxsim
        import onnx as onnx_lib
        model_raw = onnx_lib.load(str(save_path))
        model_simplified, check_ok = onnxsim.simplify(model_raw)
        if check_ok:
            onnx_lib.save(model_simplified, str(save_path))
            n_before = len(model_raw.graph.node)
            n_after  = len(model_simplified.graph.node)
            print(f"[onnxsim] {n_before} nodes → {n_after} nodes "
                  f"(-{n_before - n_after}, {(1 - n_after/n_before)*100:.0f}%)")
        else:
            print("[onnxsim] CẢNH BÁO: simplify check thất bại, giữ nguyên bản chưa simplify.")
    except ImportError:
        print("[onnxsim] chưa cài — bỏ qua (pip install onnxsim --break-system-packages "
              "để giảm node thừa trong ONNX graph).")

    # ── Validate: logits PyTorch deploy vs ONNX runtime phải khớp nhau ──
    if VALIDATE:
        _validate_onnx(save_path, model, INPUT_SIZE, ATOL, RTOL, N_SAMPLES)

    print(f"[export] size: {save_path.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()