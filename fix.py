"""export_onnx.py — Tự động reparameterize checkpoint train dở và xuất ONNX deploy."""

import torch
import torch.nn as nn
from pathlib import Path
from glkanet.builder import build_from_yaml


# ── SỬA CÁC ĐƯỜNG DẪN CHO ĐÚNG VỚI MÁY BẠN ──────────────────
PT_PATH    = "runs/exp1/best_f1.pt"   # Đường dẫn tới file train dở của bạn
YAML_PATH  = "glkanet/configs/shuffle_glkav2.yaml"  # File config kiến trúc
SAVE_PATH  = "runs/exp1/weights/best_f1_deployed.onnx" # Bản này sẽ có FPS tối đa
INPUT_SIZE = 224
OPSET      = 18
# ─────────────────────────────────────────────────────────────

class _Wrapper(nn.Module):
    """Chỉ trace logits — ONNX không hỗ trợ tuple output tốt."""
    def __init__(self, m): 
        super().__init__()
        self.m = m
    def forward(self, x): 
        return self.m(x)[0]

def main():
    print(f"[1/4] Khởi tạo cấu trúc mạng từ yaml...")
    model = build_from_yaml(YAML_PATH)
    
    print(f"[2/4] Load trọng số train dở từ {PT_PATH}...")
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    
    # Tự động bóc tách state_dict 
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    
    print(f"[3/4] ⚡ Đang thực hiện Reparameterize (Fusion các nhánh về bản Deploy)...")
    model.eval()
    # Gọi hàm tự chuyển đổi cấu trúc mạng ở đây
    model.switch_to_deploy() 

    # Đóng gói vào wrapper sau khi đã deploy
    wrapper = _Wrapper(model).cpu()
    wrapper.eval()

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    dynamic_axes = {
        "images": {0: "batch"},
        "logits": {0: "batch"},
    }

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] Đang trace model sang ONNX...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(save_path),
        opset_version=max(OPSET, 18),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"\n[Thành công] Đã reparameterize và xuất ONNX thành công!")
    print(f" └─ ONNX path: {save_path}")
    print(f" └─ Dung lượng: {save_path.stat().st_size/1024/1024:.2f} MB")
    print(f" └─ Trạng thái: Đã deploy hoàn toàn, sẵn sàng đem đi test FPS chính xác!")

if __name__ == "__main__":
    main()