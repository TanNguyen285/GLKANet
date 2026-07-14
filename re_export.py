
import inspect
from pathlib import Path
import torch

from glkanet import GLKA
from glkanet.exporter import export_all

# ── Sửa đúng đường dẫn của bạn ───────────────────────────────
ROOT        = Path(r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_augument")
CKPT_PATH   = ROOT / "best_f1.pt"                              # checkpoint gốc từ Trainer
CFG_YAML    = ROOT / "weights" / "dualattention_glkaV1.yaml"   # nếu đã bị xoá, trỏ về file yaml gốc lúc train
NUM_CLASSES = 22   # CCMT: 22 class

INPUT_SIZE = 224
OPSET      = 18
TEST_DIR       = r"C:\Users\ThisPC\Desktop\Dataset for Crop Pest and Disease Detection\CCMT Dataset-Augmented"
TEST_SPLIT_NAME  = "test_set"
CALIB_SPLIT_NAME = "train_set"

DATASET_YAML   = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\glkanet\configs\dataset.yaml"
CLASS_NAMES    = None   # để None nếu để code tự sort tên class; PHẢI khớp thứ tự lúc train nếu điền tay
TFLITE_PROJECT_DIR = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\TFlite"  # nơi chứa environment.py + bench_dataset.py
TFLITE_MODE    = "all"   # "fp32" | "fp16" | "int8" | "all"
N_CALIB = 0.05
MIXED_INT8          = True
MIXED_INT8_KEYWORDS = ["MAX_POOL_2D"]


def _call_export_all_safely(**kwargs):
    """Gọi export_all() nhưng chỉ truyền các kwarg mà chữ ký hàm THẬT SỰ chấp
    nhận — tránh crash nếu exporter.py bản đang cài chưa hỗ trợ
    test_split_name/calib_split_name/mixed_precision_int8/... (bản cũ chưa
    update tương ứng với onnx_to_tflite.py mới). Sẽ in cảnh báo rõ ràng nếu
    phải bỏ tham số nào.
    """
    sig = inspect.signature(export_all)
    accepted = set(sig.parameters.keys())
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    dropped = []
    if not accepts_var_kw:
        for k in list(kwargs.keys()):
            if k not in accepted:
                dropped.append(k)
                kwargs.pop(k)

    if dropped:
        print(f"  [!] export_all() (bản hiện tại trong exporter.py) KHÔNG nhận các "
              f"tham số: {dropped}")
        print(f"      → các tham số này bị bỏ qua, có thể ảnh hưởng tới việc chọn "
              f"đúng split khi benchmark/calib trên dataset dạng CCMT Augmented, "
              f"hoặc khiến MIXED_INT8 không có tác dụng gì.")
        print(f"      → nếu cần, cập nhật exporter.py để forward "
              f"test_split_name/calib_split_name/n_calib/mixed_precision_int8/"
              f"mixed_precision_keywords xuống onnx_to_tflite.py (đã hỗ trợ sẵn "
              f"các tham số này).")

    return export_all(**kwargs)


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
              f"accuracy nhanh, không có ảnh calib cho int8, và MIXED_INT8 sẽ tự "
              f"bị tắt (cần ảnh calib thật để chạy QuantizationDebugger).")
    if not dataset_yaml_ok:
        print(f"  [!] DATASET_YAML không tồn tại: {DATASET_YAML} — sẽ bỏ qua "
              f"bench_dataset.py (không có FPS/latency thật trên toàn bộ test_set).")

    mixed_int8_effective = MIXED_INT8 and test_dir_ok
    if MIXED_INT8 and not test_dir_ok:
        print("  [!] MIXED_INT8=True nhưng TEST_DIR không tồn tại -> tắt tự động.")

    print("\n[re_export] Bắt đầu export lại toàn bộ (train/train_full/deploy/deploy_full/onnx/tflite)...")
    print(f"[re_export] Calib int8: {N_CALIB*100:.0f}% ảnh MỖI class, "
          f"lấy từ split '{CALIB_SPLIT_NAME}' (stratified).")
    if mixed_int8_effective:
        print(f"[re_export] Mixed-precision int8 BẬT — giữ float32 cho node khớp "
              f"{MIXED_INT8_KEYWORDS} (vd MaxPool2D trong channel attention).")
    result = _call_export_all_safely(
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
        n_calib       = N_CALIB,   # FIX: giờ là fraction 0.0-1.0 (%/class), không phải số ảnh cố định
        # 2 tham số dưới — chỉ có tác dụng nếu exporter.py đã cập nhật để
        # forward xuống onnx_to_tflite.py. Nếu exporter.py chưa hỗ trợ,
        # _call_export_all_safely sẽ tự bỏ qua + in cảnh báo, không crash.
        test_split_name  = TEST_SPLIT_NAME,
        calib_split_name = CALIB_SPLIT_NAME,
        # Mixed-precision int8 — cũng chỉ có tác dụng nếu exporter.py forward
        # xuống onnx_to_tflite.convert(mixed_precision_int8=..., mixed_precision_keywords=...).
        mixed_precision_int8      = mixed_int8_effective,
        mixed_precision_keywords  = MIXED_INT8_KEYWORDS,
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