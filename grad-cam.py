# -*- coding: utf-8 -*-
"""
gradcam_compare.py
So sánh Grad-CAM giữa 2 model:
    - Model A: kernel lớn (7x7/5x5 ở tầng giữa)  -> dual_ccmt/weights/best_deploy_full.pt
    - Model B: full 3x3                          -> dual_ccmt_3x3/weights/best_deploy_full.pt

Mỗi class lấy 1 ảnh đại diện trong thư mục test/<class>/, chạy Grad-CAM cho cả
2 model, ghép: [Ảnh gốc | CAM Model A | CAM Model B] thành 1 hàng, lưu thành
1 ảnh lớn (grid) để so sánh trực quan.

Cài đặt trước khi chạy:
    pip install grad-cam opencv-python pillow pyyaml torch torchvision

Chạy:
    python gradcam_compare.py
"""

import os
import sys
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Cho phép import builder.py từ repo ─────────────────────────
REPO_ROOT = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "glkanet"))

from glkanet.builder import build_from_yaml  # noqa: E402

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ══════════════════════════════════════════════════════════════
# ⚠️ CẤU HÌNH — chỉnh lại đường dẫn nếu cần
# ══════════════════════════════════════════════════════════════

DATASET_YAML = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\glkanet\configs\dataset.yaml"

MODEL_A_NAME = "KernelLon_7x7_5x5"
MODEL_A_WEIGHTS = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt\weights\best_train.pt"
MODEL_A_YAML = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt\weights\dualattention_glkaV1.yaml"

MODEL_B_NAME = "Full_3x3"
MODEL_B_WEIGHTS = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_3x3\weights\best_train.pt"
MODEL_B_YAML = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_3x3\weights\dualattention_glkaV1.yaml"

# ⚠️ Nếu runs\dual_ccmt\weights\ KHÔNG có best_train.pt (chỉ dual_ccmt_3x3 có),
# đổi tạm MODEL_A_WEIGHTS về best_deploy_full.pt và báo lại để xử lý riêng.

OUTPUT_DIR = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\gradcam_compare_out"

IMG_SIZE = 224           # ⚠️ chỉnh lại nếu train ở size khác
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def load_model(weights_path: str, yaml_path: str) -> nn.Module:
    """Build model theo yaml rồi load weights. Tự nhận diện 3 kiểu file .pt:
        1. TorchScript archive (torch.jit.save)      -> torch.jit.load
        2. Pickle nguyên cả nn.Module object          -> torch.load trả về Module
        3. state_dict thuần (hoặc bọc trong dict)     -> build_from_yaml rồi load_state_dict
    """
    # ── Thử TorchScript trước (đây là trường hợp của best_deploy_full.pt) ──
    try:
        model = torch.jit.load(weights_path, map_location="cpu")
        model.to(DEVICE).eval()
        print("  -> Nhận diện: TorchScript archive (torch.jit.load OK)")
        return model
    except Exception:
        pass

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, nn.Module):
        print("  -> Nhận diện: pickle nguyên nn.Module")
        model = ckpt
        model.to(DEVICE).eval()
        return model

    # checkpoint dạng dict từ quá trình training thường bọc state_dict trong
    # 1 trong các key phổ biến sau; dò lần lượt cho chắc.
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                print(f"  -> Tìm thấy state_dict trong checkpoint['{key}']")
                break
        if state_dict is None:
            # có thể bản thân ckpt đã LÀ state_dict (key = tên layer, value = tensor)
            sample_keys = list(ckpt.keys())[:5]
            looks_like_state_dict = all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]) if ckpt else False
            if looks_like_state_dict:
                state_dict = ckpt
                print("  -> checkpoint chính là state_dict thuần")
            else:
                print(f"  [!] Không chắc cấu trúc checkpoint. Các key top-level: {sample_keys}")
                print(f"  [!] Toàn bộ keys: {list(ckpt.keys())}")
                raise RuntimeError(
                    "Không tự nhận diện được state_dict trong checkpoint dict. "
                    "Xem danh sách key in ở trên và sửa lại load_model() cho đúng key."
                )
    else:
        state_dict = ckpt

    print("  -> Nhận diện: state_dict, build model từ yaml rồi load weights")
    model = build_from_yaml(yaml_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e1:
        print(f"  [!] load_state_dict strict=True lỗi ({e1}); thử switch_to_deploy() rồi load lại...")
        try:
            model.switch_to_deploy()
            model.load_state_dict(state_dict, strict=True)
        except Exception as e2:
            print(f"  [!] Vẫn lỗi strict=True ({e2}); thử strict=False (kiểm tra kỹ output!)")
            model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE).eval()
    return model


def get_target_layer(model: nn.Module):
    """Lấy layer cuối cùng của backbone (ngay trước GAP trong head) làm target
    cho Grad-CAM. Hỗ trợ cả model GLKANet gốc lẫn bản đã bị torch.jit.script
    (RecursiveScriptModule, không còn attribute backbone_layers trực tiếp)."""

    # Trường hợp 1: model gốc, còn attribute backbone_layers
    if hasattr(model, "backbone_layers"):
        return model.backbone_layers[-1]

    # Trường hợp 2: model đã script hoá -> named_modules() vẫn giữ tên
    # dạng "backbone_layers.0", "backbone_layers.1", ... vì tên attribute
    # được giữ nguyên khi torch.jit.script biên dịch.
    candidates = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "backbone_layers" and parts[1].isdigit():
            candidates.append((int(parts[1]), name, module))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best = candidates[-1]
        print(f"  -> Target layer (script) tự dò được: '{best[1]}' ({best[2].__class__.__name__})")
        return best[2]

    # Trường hợp 3 (fallback cuối): lấy Conv2d cuối cùng không nằm trong head
    conv_modules = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and not name.startswith("head")
    ]
    if conv_modules:
        name, m = conv_modules[-1]
        print(f"  -> Target layer (fallback Conv2d cuối) tự dò được: '{name}'")
        return m

    # In cấu trúc model ra để người dùng tự chọn nếu vẫn không tìm được
    print("  [!] KHÔNG tìm được target layer tự động. Cấu trúc model:")
    for name, module in model.named_modules():
        print(f"      {name}: {module.__class__.__name__}")
    raise RuntimeError(
        "Không tự dò được target layer cho Grad-CAM. "
        "Hãy xem cấu trúc model in ở trên và chỉ định thủ công trong get_target_layer()."
    )


def build_class_list(dataset_root: str, test_subdir: str) -> list[str]:
    test_dir = Path(dataset_root) / test_subdir
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    return classes


def get_first_image(test_dir: Path, class_name: str) -> Path | None:
    class_dir = test_dir / class_name
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for f in sorted(class_dir.iterdir()):
        if f.suffix.lower() in exts:
            return f
    return None


def preprocess(img_pil: Image.Image):
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    tensor = tfm(img_pil).unsqueeze(0)  # (1, C, H, W)

    # ảnh rgb float [0,1] để overlay CAM lên, cùng size với tensor input
    rgb_float = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return tensor, rgb_float


class ModelWrapperForCAM(nn.Module):
    """pytorch-grad-cam cần model.forward() trả về logits (tensor), còn
    GLKANet.forward() trả về (logits, features) -> wrap lại cho gọn."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def run_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_class: int):
    wrapped = ModelWrapperForCAM(model)
    target_layer = get_target_layer(model)

    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor.to(DEVICE), targets=targets)
    grayscale_cam = grayscale_cam[0]  # (H, W)
    return grayscale_cam


def make_row(class_name: str, orig_rgb: np.ndarray, cam_a: np.ndarray, cam_b: np.ndarray,
             label_a: str, label_b: str) -> np.ndarray:
    """Ghép [Ảnh gốc | CAM A | CAM B] thành 1 hàng ảnh BGR uint8, có nhãn chữ."""
    vis_orig = (orig_rgb * 255).astype(np.uint8)
    vis_a = show_cam_on_image(orig_rgb, cam_a, use_rgb=True)
    vis_b = show_cam_on_image(orig_rgb, cam_b, use_rgb=True)

    pad_top = 30
    def add_title(img, title):
        h, w = img.shape[:2]
        canvas = np.full((h + pad_top, w, 3), 255, dtype=np.uint8)
        canvas[pad_top:, :, :] = img
        cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    vis_orig = add_title(vis_orig, "Original")
    vis_a = add_title(vis_a, label_a)
    vis_b = add_title(vis_b, label_b)

    row = np.concatenate([vis_orig, vis_a, vis_b], axis=1)

    # thanh tiêu đề class ở trên cùng hàng
    band_h = 28
    band = np.full((band_h, row.shape[1], 3), 240, dtype=np.uint8)
    cv2.putText(band, class_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    row = np.concatenate([band, row], axis=0)
    return cv2.cvtColor(row, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    dataset_root = ds_cfg["path"]
    test_subdir = ds_cfg["test"]
    test_dir = Path(dataset_root) / test_subdir

    classes = build_class_list(dataset_root, test_subdir)
    print(f"Tìm thấy {len(classes)} class trong {test_dir}")

    print(f"\n[Load] Model A ({MODEL_A_NAME}) từ {MODEL_A_WEIGHTS}")
    model_a = load_model(MODEL_A_WEIGHTS, MODEL_A_YAML)

    print(f"[Load] Model B ({MODEL_B_NAME}) từ {MODEL_B_WEIGHTS}")
    model_b = load_model(MODEL_B_WEIGHTS, MODEL_B_YAML)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for idx, class_name in enumerate(classes):
        img_path = get_first_image(test_dir, class_name)
        if img_path is None:
            print(f"  [!] Không tìm thấy ảnh cho class '{class_name}', bỏ qua.")
            continue

        img_pil = Image.open(img_path).convert("RGB")
        input_tensor, rgb_float = preprocess(img_pil)

        cam_a = run_gradcam(model_a, input_tensor, target_class=idx)
        cam_b = run_gradcam(model_b, input_tensor, target_class=idx)

        row_img = make_row(class_name, rgb_float, cam_a, cam_b, MODEL_A_NAME, MODEL_B_NAME)
        rows.append(row_img)

        # lưu riêng từng class luôn, để dễ xem/so sánh lẻ
        out_path = os.path.join(OUTPUT_DIR, f"{idx:02d}_{class_name.replace('/', '_')}.png")
        cv2.imwrite(out_path, row_img)
        print(f"  [{idx+1}/{len(classes)}] {class_name} -> {out_path}")

    # ghép tất cả thành 1 ảnh tổng (grid dọc)
    if rows:
        # đảm bảo cùng width trước khi concat dọc
        max_w = max(r.shape[1] for r in rows)
        rows_padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            rows_padded.append(r)
        full_grid = np.concatenate(rows_padded, axis=0)
        full_path = os.path.join(OUTPUT_DIR, "_ALL_CLASSES_gradcam_compare.png")
        cv2.imwrite(full_path, full_grid)
        print(f"\n✅ Đã lưu ảnh tổng hợp toàn bộ {len(rows)} class: {full_path}")


if __name__ == "__main__":
    main()