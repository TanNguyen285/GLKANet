"""glkanet/onnx_to_tflite.py — Convert best_deploy.onnx -> TFLite (fp32/fp16/int8)
để test FPS trên Raspberry Pi, và eval accuracy ONNX + TFLite trên test set thật.

=====================================================================
CÀI ĐẶT (Python 3.10.11) — dùng venv riêng, KHÔNG cài chung venv train:

    python -m venv venv-tflite
    venv-tflite\\Scripts\\activate          (Windows)
    source venv-tflite/bin/activate         (Linux/Mac)

    pip install -r requirements-tflite.txt

=====================================================================
CÁCH DÙNG:

    python onnx_to_tflite.py --onnx weights/best_deploy.onnx --out weights/tflite \
        --input-size 224 --mode all --calib-dir dataset/test --n-calib 200 \
        --test-dir dataset/test --class-names class0,class1,class2 \
        --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225

    --mode fp32   -> chỉ lấy model_float32.tflite
    --mode fp16   -> chỉ lấy model_float16.tflite
    --mode int8   -> chỉ lấy model_integer_quant.tflite (BẮT BUỘC --calib-dir)
    --mode all    -> lấy hết (mặc định)

    --test-dir    -> nếu truyền vào, sau khi convert xong sẽ tự eval accuracy
                     ONNX + mọi file .tflite sinh ra, trên đúng ảnh trong thư mục này
                     (ImageFolder: test_dir/class_name/*.jpg), ghi kết quả ra json.

    --mean/--std  -> PHẢI khớp đúng transforms.Normalize() lúc train (xem train.yaml
                     mục normalize:). Mặc định ImageNet [0.485,0.456,0.406] /
                     [0.229,0.224,0.225] nếu không truyền.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


REQUIRED_PACKAGES = ("onnx", "onnx2tf", "tensorflow", "onnxsim")

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)


def _check_deps() -> None:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception:
            print(f"[LỖI IMPORT] '{pkg}' cài rồi nhưng import lỗi:")
            import traceback
            traceback.print_exc()
            missing.append(pkg)
    if missing:
        print(f"\n[LỖI] Import thất bại: {', '.join(missing)}")
        print("      Xem traceback ở trên để biết nguyên nhân thật "
              "(thường là xung đột version, không hẳn là thiếu cài).")
        sys.exit(1)


def _get_onnx_input_name(onnx_path: Path) -> str:
    import onnx as onnx_lib
    model = onnx_lib.load(str(onnx_path))
    return model.graph.input[0].name


def _collect_calib_images(calib_dir: str, n_calib: int) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    calib_path = Path(calib_dir)
    if not calib_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục calib: {calib_dir}")
    img_paths = sorted(p for p in calib_path.rglob("*") if p.suffix.lower() in exts)
    if not img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {calib_dir}")
    return img_paths[:n_calib]


def _build_calibration_npy(
    calib_dir: str,
    input_size: int,
    n_calib: int,
    out_dir: Path,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> Path:
    """Đọc ảnh thật, resize + normalize ĐÚNG như lúc train (mean/std ImageNet mặc định),
    gộp thành 1 file .npy shape (N, H, W, 3) float32 — format onnx2tf cần cho quant int8.

    QUAN TRỌNG: mean/std truyền vào đây PHẢI khớp transforms.Normalize() lúc train,
    nếu không phân phối dữ liệu calibrate sẽ sai lệch, kéo theo quant int8 sai range.
    """
    from PIL import Image

    img_paths = _collect_calib_images(calib_dir, n_calib)
    print(f"[calib] Dùng {len(img_paths)} ảnh thật từ {calib_dir} để calibrate int8 "
          f"(mean={mean}, std={std}).")

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    arrs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB").resize((input_size, input_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean_arr) / std_arr
        arrs.append(arr)

    data = np.stack(arrs, axis=0)  # (N, H, W, 3)
    npy_path = out_dir / "calib_data.npy"
    np.save(npy_path, data)
    return npy_path


def convert(
    onnx_path: str,
    out_dir: str,
    input_size: int = 224,
    mode: str = "all",  # "fp32" | "fp16" | "int8" | "all"
    calib_dir: str | None = None,
    n_calib: int = 200,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> None:
    _check_deps()
    import onnx2tf

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {onnx_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = out_dir / "saved_model"

    need_int8 = mode in ("int8", "all")

    print(f"[1/2] Convert ONNX -> TF SavedModel + TFLite (onnx2tf): {onnx_path.name}")

    legacy_cache_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    legacy_cache_path = Path.cwd() / legacy_cache_name
    if not legacy_cache_path.exists():
        dummy_legacy = np.random.rand(20, 128, 128, 3).astype(np.float32)
        np.save(legacy_cache_path, dummy_legacy)

    test_data_path = out_dir / "_dummy_test_data.npy"
    if not test_data_path.exists():
        dummy = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
        np.save(test_data_path, dummy)

    kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        test_data_nhwc_path=str(test_data_path),
    )

    if need_int8:
        if calib_dir:
            input_name = _get_onnx_input_name(onnx_path)
            npy_path = _build_calibration_npy(
                calib_dir, input_size, n_calib, out_dir, mean=mean, std=std,
            )
            # Ảnh trong npy đã normalize (mean/std) sẵn -> báo onnx2tf mean=0, std=1
            # để nó KHÔNG normalize thêm lần nữa (tránh normalize 2 lần chồng nhau).
            kwargs["custom_input_op_name_np_data_path"] = [
                [input_name, str(npy_path), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            ]
            kwargs["output_integer_quantized_tflite"] = True
        else:
            print("[INFO] Không có --calib-dir -> dùng Dynamic Range Quantization "
                  "(int8 weight-only, không cần ảnh calib).")
            kwargs["output_dynamic_range_quantized_tflite"] = True

    # =====================================================================
    # ĐOẠN SỬA ĐƯỜNG DẪN AUTO JSON ĐỂ KÍCH HOẠT RETRY:
    # =====================================================================
    # SỬA TẠI ĐÂY: onnx2tf sinh file JSON nằm TRONG thư mục saved_model
    auto_json_path = saved_model_dir / f"{onnx_path.stem}_auto.json"

    def _do_convert(**kw):
        onnx2tf.convert(**kw)

    try:
        _do_convert(**kwargs)
    except TypeError as e:
        if "test_data_nhwc_path" in str(e):
            print("[CẢNH BÁO] Bản onnx2tf này không nhận tham số test_data_nhwc_path "
                  "-> bỏ tham số đó, thử lại.")
            kwargs.pop("test_data_nhwc_path", None)
            _do_convert(**kwargs)
        else:
            raise
    except Exception as e:
        # onnx2tf tự sinh file sửa layout tại saved_model_dir / f"{onnx_path.stem}_auto.json"
        if auto_json_path.exists():
            print(f"\n[RETRY] Convert lỗi ({e.__class__.__name__}). onnx2tf đã tự sinh "
                  f"file sửa layout tại:\n         {auto_json_path}\n"
                  f"         -> Thử convert lại với param_replacement_file...")
            kwargs["param_replacement_file"] = str(auto_json_path)
            try:
                _do_convert(**kwargs)
                print("[RETRY] Convert lại thành công với auto JSON!")
            except Exception as e2:
                print(f"[LỖI] Convert lại với auto JSON vẫn thất bại: {e2}")
                print(f"      Cần sửa thủ công file JSON tại {auto_json_path} — "
                      f"xem hướng dẫn: https://github.com/PINTO0309/onnx2tf#parameter-replacement")
                raise
        else:
            print(f"[THÔNG BÁO] Không tìm thấy file JSON tự động tại: {auto_json_path}")
            raise
    # =====================================================================

    print("[2/2] Sắp xếp lại file .tflite ra out_dir gốc, dễ copy sang Pi")
    produced = list(saved_model_dir.glob("*.tflite"))
    if not produced:
        print("[CẢNH BÁO] Không thấy file .tflite nào được sinh ra — kiểm tra log onnx2tf ở trên.")
        return

    keep_keywords = {
        "fp32": ("float32",),
        "fp16": ("float16",),
        "int8": ("integer_quant", "dynamic_range_quant", "int8"),
        "all": ("float32", "float16", "integer_quant", "dynamic_range_quant", "int8"),
    }[mode]

    copied = []
    for f in produced:
        if any(k in f.name for k in keep_keywords):
            dst = out_dir / f.name
            shutil.copy2(f, dst)
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  -> {dst.name:<40} {size_mb:.2f} MB")
            copied.append(dst)

    if not copied:
        print("[CẢNH BÁO] Không có file nào khớp --mode, kiểm tra lại tên file trong "
              f"{saved_model_dir}")
        return

    print(f"\nXong. Các file .tflite nằm trong '{out_dir}'.")


# ═══════════════════════════════════════════════════════════════════════
#  Eval accuracy ONNX + TFLite trên test set thật (chạy trong venv-tflite)
# ═══════════════════════════════════════════════════════════════════════

def _list_class_dirs(test_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(test_dir).iterdir() if p.is_dir())


def _load_test_set(
    test_dir: str,
    input_size: int,
    class_names: list[str] | None = None,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    """Đọc toàn bộ ảnh test thật từ thư mục dạng ImageFolder (class_name/*.jpg),
    normalize ĐÚNG mean/std lúc train (mặc định ImageNet). Trả về ảnh đã normalize
    dạng NHWC float32 — dùng chung cho cả ONNX (transpose sang NCHW sau) và TFLite.
    """
    from PIL import Image

    test_dir = Path(test_dir)
    classes = class_names or _list_class_dirs(test_dir)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    images, labels = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for c in classes:
        cdir = test_dir / c
        if not cdir.exists():
            continue
        for p in sorted(cdir.rglob("*")):
            if p.suffix.lower() not in exts:
                continue
            img = Image.open(p).convert("RGB").resize((input_size, input_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - mean_arr) / std_arr
            images.append(arr)
            labels.append(cls_to_idx[c])

    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {test_dir}")

    return np.stack(images, axis=0), np.array(labels, dtype=np.int64), classes


def _eval_onnx_backend(onnx_path: Path, images_nhwc: np.ndarray, labels: np.ndarray) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # ONNX wrapper export dùng NCHW (xem exporter.py: dummy = [1,3,H,W])
    images_nchw = np.transpose(images_nhwc, (0, 3, 1, 2)).astype(np.float32)

    correct, total = 0, 0
    t0 = time.perf_counter()
    for i in range(images_nchw.shape[0]):
        logits = sess.run(None, {input_name: images_nchw[i:i + 1]})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        correct += int(pred == labels[i])
        total += 1
    elapsed = time.perf_counter() - t0
    return {"acc": correct / total, "ms_per_img": elapsed / max(total, 1) * 1000}


def _eval_tflite_backend(tflite_path: Path, images_nhwc: np.ndarray, labels: np.ndarray) -> dict:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # images_nhwc đã normalize (mean/std) sẵn -> nếu input model là float thì dùng
    # thẳng; nếu model int8 (full-integer quant) thì cần quantize lại theo scale/
    # zero_point riêng của input đó (KHÁC với normalize ảnh, đây là bước lượng tử
    # hoá thêm sau khi đã normalize).
    x = images_nhwc
    if input_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        x = (x / scale + zero_point).round().astype(input_details["dtype"])
    else:
        x = x.astype(np.float32)

    correct, total = 0, 0
    t0 = time.perf_counter()
    for i in range(x.shape[0]):
        interpreter.resize_tensor_input(input_details["index"], (1, *x.shape[1:]))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details["index"], x[i:i + 1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] in (np.int8, np.uint8):
            scale, zero_point = output_details["quantization"]
            out = (out.astype(np.float32) - zero_point) * scale
        pred = int(np.argmax(out, axis=1)[0])
        correct += int(pred == labels[i])
        total += 1
    elapsed = time.perf_counter() - t0
    return {"acc": correct / total, "ms_per_img": elapsed / max(total, 1) * 1000}


def evaluate_backends(
    onnx_path: str,
    tflite_dir: str,
    test_dir: str,
    input_size: int,
    class_names: list[str] | None,
    eval_out_json: str,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> dict:
    """Chạy accuracy test trên ONNX + mọi file .tflite trong tflite_dir, cùng 1 test set thật.
    Ghi kết quả ra eval_out_json để process train (có torch) đọc lại."""
    print(f"[eval] Đang load test set từ {test_dir} (mean={mean}, std={std}) ...")
    images_nhwc, labels, classes = _load_test_set(test_dir, input_size, class_names, mean=mean, std=std)
    print(f"[eval] {len(labels)} ảnh, {len(classes)} class.")

    results: dict[str, dict] = {}

    onnx_path = Path(onnx_path)
    if onnx_path.exists():
        print("[eval] Đang test ONNX Runtime...")
        results["onnx"] = _eval_onnx_backend(onnx_path, images_nhwc, labels)
        print(f"       acc={results['onnx']['acc']*100:.2f}%  {results['onnx']['ms_per_img']:.2f} ms/ảnh")

    for tflite_path in sorted(Path(tflite_dir).glob("*.tflite")):
        key = tflite_path.stem
        print(f"[eval] Đang test TFLite [{key}]...")
        try:
            results[key] = _eval_tflite_backend(tflite_path, images_nhwc, labels)
            print(f"       acc={results[key]['acc']*100:.2f}%  {results[key]['ms_per_img']:.2f} ms/ảnh")
        except Exception as e:  # noqa: BLE001
            print(f"       LỖI khi test {tflite_path.name}: {e}")
            results[key] = {"error": str(e)}

    Path(eval_out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[eval] Đã ghi kết quả -> {eval_out_json}")
    return results


def _parse_mean_std(s: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not s:
        return default
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"mean/std phải có đúng 3 giá trị cách nhau bởi dấu phẩy, nhận: {s}")
    return tuple(parts)  # type: ignore[return-value]


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Convert ONNX -> TFLite (fp32/fp16/int8) + eval accuracy")
    ap.add_argument("--onnx", required=True, help="Path tới best_deploy.onnx")
    ap.add_argument("--out", default="weights/tflite", help="Thư mục xuất .tflite")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--mode", choices=["fp32", "fp16", "int8", "all"], default="all")
    ap.add_argument("--calib-dir", default=None,
                     help="Thư mục ảnh thật để calibrate int8 (bắt buộc nếu mode=int8/all)")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--test-dir", default=None,
                     help="Thư mục test THẬT (ImageFolder: class_name/*.jpg) để eval accuracy sau convert")
    ap.add_argument("--class-names", default=None,
                     help="Danh sách class cách nhau bởi dấu phẩy, PHẢI đúng thứ tự index lúc train")
    ap.add_argument("--eval-out", default=None,
                     help="Path json ghi kết quả eval (mặc định: <out>/backend_eval_results.json)")
    ap.add_argument("--mean", default=None,
                     help="mean normalize, 3 số cách nhau dấu phẩy, VD: 0.485,0.456,0.406 "
                          "(PHẢI khớp train.yaml -> normalize.mean, mặc định ImageNet nếu bỏ trống)")
    ap.add_argument("--std", default=None,
                     help="std normalize, 3 số cách nhau dấu phẩy, VD: 0.229,0.224,0.225 "
                          "(PHẢI khớp train.yaml -> normalize.std, mặc định ImageNet nếu bỏ trống)")
    args = ap.parse_args()

    mean = _parse_mean_std(args.mean, DEFAULT_MEAN)
    std  = _parse_mean_std(args.std, DEFAULT_STD)

    convert(
        onnx_path=args.onnx,
        out_dir=args.out,
        input_size=args.input_size,
        mode=args.mode,
        calib_dir=args.calib_dir,
        n_calib=args.n_calib,
        mean=mean,
        std=std,
    )

    if args.test_dir:
        class_names = args.class_names.split(",") if args.class_names else None
        eval_out = args.eval_out or str(Path(args.out) / "backend_eval_results.json")
        evaluate_backends(
            onnx_path=args.onnx,
            tflite_dir=args.out,
            test_dir=args.test_dir,
            input_size=args.input_size,
            class_names=class_names,
            eval_out_json=eval_out,
            mean=mean,
            std=std,
        )


if __name__ == "__main__":
    _cli()