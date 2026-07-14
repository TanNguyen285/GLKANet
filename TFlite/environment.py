"""environment.py — Chạy 1 phát:
  - Lần ĐẦU: chưa có venv -> tự tạo venv-tflite, cài đúng bộ thư viện đã pin, xong tự convert luôn.
  - Các lần SAU: venv đã có sẵn -> tự "join" (dùng lại) venv đó, KHÔNG cài lại, convert ngay ra .tflite.

Cách dùng (CLI, để trainer.py/exporter.py gọi tự động qua subprocess):
    python environment.py --onnx path/to/best_deploy.onnx --out path/to/tflite \
        --input-size 224 --mode all --calib-dir path/to/dataset --test-dir path/to/dataset \
        --calib-split-name train_set --test-split-name test_set \
        --class-names c0,c1,c2 --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225

File này cần nằm CÙNG THƯ MỤC với:
  - onnx_to_tflite.py
  - requirements-tflite.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / "venv-tflite"
REQ_FILE = HERE / "requirements-tflite.txt"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
INSTALL_MARKER = VENV_DIR / ".install_ok"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--mode", default="all", choices=["fp32", "fp16", "int8", "all"])
    ap.add_argument("--calib-dir", default=None)
    # FIX: n_calib giờ là TỈ LỆ (fraction) ảnh mỗi class dùng để calibrate
    # int8, KHÔNG còn là tổng số ảnh cố định (200) như bản cũ. Khớp với
    # onnx_to_tflite.py đã sửa (_collect_calib_images lấy đều %/class).
    # 0.10 = 10% ảnh mỗi class. Tăng lên 0.2 nếu 10% chưa đủ ổn định accuracy.
    ap.add_argument("--n-calib", type=float, default=0.20,)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--class-names", default=None)
    ap.add_argument("--eval-out", default=None)
    # FIX: thiếu 2 tham số này ở bản cũ khiến calib_split_name/test_split_name
    # bị rớt mất khi environment.py forward xuống onnx_to_tflite.py — dù
    # re_export.py có set CALIB_SPLIT_NAME="train_set" thì cũng không tới
    # được đây, convert() sẽ âm thầm dùng mặc định "test_set" bên trong.
    ap.add_argument("--calib-split-name", default="test_set",
                     help="Split dùng để lấy ảnh calib int8 (CCMT 2 tầng), "
                          "vd 'train_set' hoặc 'test_set'. Bỏ qua nếu calib-dir "
                          "là ImageFolder phẳng.")
    ap.add_argument("--test-split-name", default="test_set",
                     help="Split dùng để eval accuracy sau convert (CCMT 2 tầng). "
                          "Bỏ qua nếu test-dir là ImageFolder phẳng.")
    ap.add_argument("--sample-fraction", type=float, default=0.10,
                     help="Tỉ lệ ảnh MỖI class dùng để eval accuracy sau convert "
                          "(mặc định 0.10 = 10%%).")
    ap.add_argument("--mean", default=None,
                     help="3 số cách nhau dấu phẩy, VD: 0.485,0.456,0.406. "
                          "Mặc định ImageNet nếu bỏ trống.")
    ap.add_argument("--std", default=None,
                     help="3 số cách nhau dấu phẩy, VD: 0.229,0.224,0.225. "
                          "Mặc định ImageNet nếu bỏ trống.")
    ap.add_argument("--mixed-int8", action="store_true",
                     help="Xuất thêm bản int8 denylist MaxPool2D (giữ float32 riêng).")
    ap.add_argument("--mixed-int8-keywords", default="MaxPool2D",
                     help="Từ khóa denylist, cách nhau dấu phẩy.")
    return ap.parse_args()


def _is_running_inside_venv() -> bool:
    if not VENV_PY.exists():
        return False
    try:
        return os.path.samefile(sys.executable, VENV_PY)
    except OSError:
        return False


def _bootstrap_venv() -> None:
    """Tạo venv (nếu chưa có) + cài thư viện (nếu chưa cài THÀNH CÔNG lần nào)."""
    if INSTALL_MARKER.exists():
        print("[OK] venv-tflite đã có sẵn và đã cài đủ thư viện -> join vào luôn.\n")
        return

    if not REQ_FILE.exists():
        print(f"[LỖI] Không tìm thấy {REQ_FILE.name} cùng thư mục với environment.py")
        sys.exit(1)

    if not VENV_PY.exists():
        print("[1/2] Chưa có môi trường -> đang tạo venv-tflite ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    else:
        print("[1/2] venv-tflite đã có nhưng chưa cài xong thư viện -> cài lại.")

    print("[2/2] Đang cài thư viện (mất vài phút, cần mạng) ...")
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "-r", str(REQ_FILE)], check=True)
    INSTALL_MARKER.write_text("ok")
    print("Cài xong môi trường!\n")


def _relaunch_inside_venv() -> int:
    """Gọi lại chính file environment.py này, nhưng bằng python NẰM TRONG venv,
    forward nguyên vẹn toàn bộ argv (trừ argv[0])."""
    cmd = [str(VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:]
    result = subprocess.run(cmd)
    return result.returncode


def _run_conversion(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(HERE))
    from onnx_to_tflite import convert, evaluate_backends, DEFAULT_MEAN, DEFAULT_STD

    def _parse_triplet(s, default):
        if not s:
            return default
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"mean/std phải có đúng 3 giá trị, nhận: {s}")
        return tuple(parts)

    mean = _parse_triplet(args.mean, DEFAULT_MEAN)
    std  = _parse_triplet(args.std, DEFAULT_STD)

    print("=== Bắt đầu convert ONNX -> TFLite ===")
    print(f"    calib: {args.n_calib*100:.0f}% ảnh/class, split='{args.calib_split_name}'")

    convert(
        onnx_path=args.onnx,
        out_dir=args.out,
        input_size=args.input_size,
        mode=args.mode,
        calib_dir=args.calib_dir,
        n_calib=args.n_calib,
        calib_split_name=args.calib_split_name,
        mean=mean,
        std=std,
        mixed_precision_int8=args.mixed_int8,
        mixed_precision_keywords=[k.strip() for k in args.mixed_int8_keywords.split(",") if k.strip()],
    
    )

    if args.test_dir:
        class_names = args.class_names.split(",") if args.class_names else None
        eval_out = args.eval_out or str(Path(args.out) / "backend_eval_results.json")
        print("=== Bắt đầu eval accuracy ONNX + TFLite trên test set thật ===")
        print(f"    eval: {args.sample_fraction*100:.0f}% ảnh/class, split='{args.test_split_name}'")
        evaluate_backends(
            onnx_path=args.onnx,
            tflite_dir=args.out,
            test_dir=args.test_dir,
            input_size=args.input_size,
            class_names=class_names,
            eval_out_json=eval_out,
            mean=mean,
            std=std,
            split_name=args.test_split_name,          # FIX: forward xuống, trước đây bị rớt
            max_samples=None,
            sample_fraction=args.sample_fraction,      # FIX: dùng %/class thay vì mặc định max_samples=1000
        )


def main() -> None:
    args = _parse_args()

    if _is_running_inside_venv():
        _run_conversion(args)
        return

    _bootstrap_venv()
    sys.exit(_relaunch_inside_venv())


if __name__ == "__main__":
    main()