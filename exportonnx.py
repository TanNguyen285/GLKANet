"""re_export.py — Load lại từ best_f1.pt (checkpoint tốt nhất theo F1, do Trainer
lưu trong runs/dual_ccmt/, KHÁC với 3 file trong weights/) và export lại TOÀN BỘ
(train/train_full/deploy/deploy_full/onnx/tflite) bằng exporter.py bản mới.

Dùng khi các file trong weights/ (best_deploy.pt, .onnx...) bị hỏng/thiếu nhưng
checkpoint gốc best_f1.pt vẫn còn nguyên.

Chạy:
    python re_export.py
"""
from pathlib import Path
import torch

from glkanet import GLKA
from glkanet.exporter import export_all

# ── Sửa đúng đường dẫn của bạn ───────────────────────────────
ROOT        = Path(r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_conv2d")
CKPT_PATH   = ROOT / "best_f1.pt"                              # checkpoint gốc từ Trainer
CFG_YAML    = ROOT / "weights" / "dualattention_glkaV1.yaml"   # nếu đã bị xoá, trỏ về file yaml gốc lúc train
NUM_CLASSES = 22   # CCMT: 22 class

INPUT_SIZE = 224
OPSET      = 18

# ── TFLite / benchmark thật ───────────────────────────────────
TEST_DIR       = r"C:\Users\ThisPC\Desktop\Raw Data\CCMT Dataset\test"         # ImageFolder ảnh test thật — SỬA lại đường dẫn thật
DATASET_YAML   = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\glkanet\configs\dataset.yaml"
CLASS_NAMES    = None   # để None nếu ImageFolder tự đọc được tên class từ tên thư mục con
TFLITE_PROJECT_DIR = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\TFlite"  # nơi chứa environment.py + bench_dataset.py
TFLITE_MODE    = "all"   # "fp32" | "fp16" | "int8" | "all"
N_CALIB        = 200


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {CKPT_PATH} — nếu file này cũng hỏng thì bắt buộc phải "
            f"train lại từ đầu, không còn cách nào khác để lấy lại weights."
        )
    if not CFG_YAML.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {CFG_YAML} — cần đúng file yaml kiến trúc đã dùng lúc "
            f"train ra checkpoint này (không phải yaml đã sửa sau đó)."
        )

    print(f"[re_export] Load lại từ: {CKPT_PATH}")
    wrapper = GLKA(str(CFG_YAML))
    model = wrapper._build_with_nc(NUM_CLASSES)

    # Checkpoint Trainer thường có nhiều key hơn (epoch, optimizer, scheduler...),
    # chỉ cần lấy đúng phần state_dict của model.
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            # Có thể bản thân ckpt đã LÀ state_dict (key = tên layer thẳng)
            state_dict = ckpt
    else:
        state_dict = ckpt

    print(f"  [i] Checkpoint có các key top-level: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [!] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"       missing (tối đa 10): {missing[:10]}")
        if unexpected:
            print(f"       unexpected (tối đa 10): {unexpected[:10]}")
        print("  [!] Nếu danh sách missing/unexpected KHÔNG rỗng, kiểm tra lại xem")
        print("      CFG_YAML có đúng là bản yaml lúc train ra checkpoint này không —")
        print("      ĐỪNG tin kết quả export nếu còn key lệch.")
    else:
        print("  [OK] load_state_dict khớp hoàn toàn, không thiếu/thừa key nào.")

    model.eval()

    # ── Kiểm tra trước khi chạy — tránh mất công đợi export xong mới báo lỗi ──
    test_dir_ok     = Path(TEST_DIR).exists() if TEST_DIR else False
    dataset_yaml_ok = Path(DATASET_YAML).exists() if DATASET_YAML else False
    tflite_dir_ok   = (Path(TFLITE_PROJECT_DIR) / "environment.py").exists()

    if not tflite_dir_ok:
        print(f"  [!] Không tìm thấy {TFLITE_PROJECT_DIR}\\environment.py — "
              f"bước TFLite sẽ tự bỏ qua bên trong exporter.py (không crash).")
    if not test_dir_ok:
        print(f"  [!] TEST_DIR không tồn tại: {TEST_DIR} — sẽ không có backend_eval "
              f"accuracy nhanh, và không có ảnh calib cho int8 (int8 vẫn chạy nhưng "
              f"dùng dữ liệu giả nếu onnx_to_tflite.py hỗ trợ, kiểm tra lại).")
    if not dataset_yaml_ok:
        print(f"  [!] DATASET_YAML không tồn tại: {DATASET_YAML} — sẽ bỏ qua "
              f"bench_dataset.py (không có FPS/latency thật trên toàn bộ test_set).")

    print("\n[re_export] Bắt đầu export lại toàn bộ (train/train_full/deploy/deploy_full/onnx/tflite)...")
    result = export_all(
        model         = model,
        save_dir      = ROOT,           # export_all tự tạo save_dir/weights/
        yaml_path     = CFG_YAML,
        input_size    = INPUT_SIZE,
        opset         = OPSET,
        verbose       = True,
        validate      = True,
        export_torchscript = True,
        export_tflite = True,
        test_dir      = TEST_DIR if test_dir_ok else None,
        dataset_yaml  = DATASET_YAML if dataset_yaml_ok else None,
        class_names   = CLASS_NAMES,
        tflite_project_dir = TFLITE_PROJECT_DIR,
        tflite_mode   = TFLITE_MODE,
        n_calib       = N_CALIB,
    )

    print("\n[re_export] Xong. Các file đã tạo:")
    for k in ("train", "train_full", "deploy", "deploy_full", "onnx", "tflite_dir", "yaml"):
        p = result.get(k)
        if p:
            print(f"  {k:<12} → {p}")
    print(f"  validated   → {result.get('validated')}")

    if result.get("test_results"):
        print("\n[re_export] test_results (torch/onnx/tflite):")
        for name, r in result["test_results"].items():
            print(f"  {name}: {r}")


if __name__ == "__main__":
    main()