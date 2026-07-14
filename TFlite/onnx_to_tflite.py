from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAM_REPLACEMENT_FILE = HERE / "param_replacement.json"

REQUIRED_PACKAGES = ("onnx", "onnx2tf", "tensorflow", "onnxsim")

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Tên file cuối cùng cố định, không lệ thuộc naming của onnx2tf.
FINAL_NAMES = {
    "fp32": "best_deploy_fp32.tflite",
    "fp16": "best_deploy_fp16.tflite",
    "int8": "best_deploy_int8.tflite",
}
# Từ khóa onnx2tf dùng để nhận diện file nó sinh ra, map sang key FINAL_NAMES.
SOURCE_KEYWORDS = {
    "fp32": ("float32",),
    "fp16": ("float16",),
    "int8": ("integer_quant", "dynamic_range_quant", "int8"),
}

# Tên file cho bản int8 mixed-precision (MaxPool/ECA giữ float32).
MIXED_INT8_NAME = "best_deploy_int8_mixedfp32.tflite"


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


def _norm_cls(name: str) -> str:
    """Bỏ số cuối tên class, vd 'anthracnose1' -> 'anthracnose' (kiểu CCMT)."""
    return re.sub(r'\d+$', '', name).strip()


# ═══════════════════════════════════════════════════════════════════════
#  Auto-detect cấu trúc dataset: ImageFolder phẳng  vs  CCMT 2 tầng
#
#  Flat (ImageFolder):   root/<class>/*.jpg
#  CCMT:                 root/<Group>/<split_name>/<class>/*.jpg
#      vd: CCMT Dataset-Augmented/Cashew/test_set/anthracnose1/*.jpg
# ═══════════════════════════════════════════════════════════════════════

def _scan_dataset(
    root: str | Path,
    split_name: str | None = None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")

    top_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not top_dirs:
        raise FileNotFoundError(f"Không có thư mục con nào trong: {root}")

    def _has_images_direct(d: Path) -> bool:
        return any(f.is_file() and f.suffix.lower() in IMG_EXTS for f in d.iterdir())

    # Heuristic: nếu >=1 top_dir chứa ảnh trực tiếp -> flat ImageFolder
    flat_like = any(_has_images_direct(d) for d in top_dirs)

    raw: list[tuple[Path, str]] = []

    if flat_like:
        # ── Flat ImageFolder: root/<class>/*.jpg ──
        for cd in top_dirs:
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    raw.append((f, cd.name))
        if not raw:
            raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {root}")
        class_names = sorted({c for _, c in raw})
        return raw, class_names

    # ── CCMT 2 tầng: root/<Group>/<split_name>/<class>/*.jpg ──
    if split_name is None:
        raise FileNotFoundError(
            f"Không phát hiện ảnh trực tiếp trong các thư mục con của {root} "
            f"(có vẻ là cấu trúc CCMT 2 tầng: Group/split/class) nhưng thiếu "
            f"split_name (vd 'test_set') để biết quét split nào."
        )

    found_any_split = False
    for group_dir in top_dirs:
        split_dir = group_dir / split_name
        if not split_dir.exists():
            continue
        found_any_split = True
        group_name = group_dir.name
        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for cd in class_dirs:
            cls = _norm_cls(cd.name)
            full_name = f"{group_name}_{cls}"
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    raw.append((f, full_name))

    if not found_any_split:
        raise FileNotFoundError(
            f"Không tìm thấy '{split_name}/' trong bất kỳ group nào tại: {root}\n"
            f"Group tìm thấy: {[d.name for d in top_dirs]}"
        )
    if not raw:
        raise FileNotFoundError(f"Không có ảnh nào trong CCMT split '{split_name}' tại: {root}")

    class_names = sorted({c for _, c in raw})
    return raw, class_names


# ═══════════════════════════════════════════════════════════════════════
#  Calibration cho int8 — stratified đều theo class.
# ═══════════════════════════════════════════════════════════════════════

def _collect_calib_images(
    calib_dir: str,
    fraction: float,
    split_name: str | None = "test_set",
    min_per_class: int = 3,
    seed: int = 42,
) -> list[Path]:
    """Lấy `fraction` ảnh MỖI class để calibrate int8 (stratified đều theo
    class, cùng logic với cách sample test-set eval).
    """
    calib_path = Path(calib_dir)
    if not calib_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục calib: {calib_dir}")

    raw, _ = _scan_dataset(calib_path, split_name=split_name)
    if not raw:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {calib_dir}")

    by_class: dict[str, list[Path]] = {}
    for p, cname in raw:
        by_class.setdefault(cname, []).append(p)

    rng = np.random.default_rng(seed)
    out: list[Path] = []
    for cname, paths in by_class.items():
        paths = list(paths)
        idx = np.arange(len(paths))
        rng.shuffle(idx)
        k = max(min_per_class, int(round(len(paths) * fraction)))
        k = min(k, len(paths))
        out.extend(paths[i] for i in idx[:k])

    print(f"[calib] Stratified {fraction*100:.0f}%/class: {len(by_class)} class, "
          f"tổng {len(out)} ảnh calib (min {min_per_class}/class, seed={seed}).")
    return out


def _build_calibration_npy(
    calib_dir: str,
    input_size: int,
    calib_fraction: float,
    out_dir: Path,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    split_name: str | None = "test_set",
) -> Path:
    """Đọc ảnh thật, resize + normalize ĐÚNG như lúc train, gộp thành 1 file .npy
    shape (N, H, W, 3) float32 — format onnx2tf cần cho quant int8.
    """
    from PIL import Image

    img_paths = _collect_calib_images(calib_dir, fraction=calib_fraction, split_name=split_name)
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


# ═══════════════════════════════════════════════════════════════════════
#  Mixed-precision int8: denylist node khớp keyword (vd MaxPool) để giữ
#  float32 riêng cho chúng, phần còn lại vẫn int8 bình thường.
#
#  Cơ chế: TFLiteConverter build trực tiếp từ saved_model onnx2tf đã sinh
#  ra (không qua lại onnx2tf lần 2), dùng tf.lite.experimental.
#  QuantizationDebugger với denylisted_nodes để loại các node khớp keyword
#  khỏi lượng tử hóa int8 — TFLite runtime sẽ tự chèn dequant/quant quanh
#  đúng những node đó, không đụng tới phần còn lại của model.
#
#  LƯU Ý: node bị denylist chạy float32, KHÔNG phải int16 thật sự — TFLite
#  không có kernel int16 chọn theo từng op riêng lẻ, chỉ có chế độ toàn
#  model "int8 + int16 activation". Nếu cần đúng nghĩa int16 cho riêng
#  MaxPool thì không có API TFLite làm việc đó ở mức per-op.
# ════

def _find_deny_nodes(saved_model_dir: Path, deny_keywords: list[str]) -> list[str]:
  
    import tensorflow as tf

    sm = tf.saved_model.load(str(saved_model_dir))
    concrete = sm.signatures["serving_default"]
    graph_def = concrete.graph.as_graph_def(add_shapes=False)

    matches: list[str] = []
    seen: set[str] = set()

    # Hàm xóa sạch ký tự gây nhiễu để đưa về chuỗi gốc (ví dụ: max_pool_2d -> maxpool)
    def _normalize(s: str) -> str:
        return s.lower().replace("_", "").replace("2d", "").replace("3d", "")

    # Chuẩn hóa toàn bộ danh sách keyword chặn đầu vào
    normalized_keywords = [_normalize(kw) for kw in deny_keywords]

    def _check_and_collect(nodes):
        for node in nodes:
            # Chuẩn hóa cả tên node và loại op thực tế của node đó
            norm_node_name = _normalize(node.name)
            norm_node_op = _normalize(node.op)
            
            # Chỉ cần keyword lọt vào tên hoặc op sau khi chuẩn hóa là dính liền
            is_match = any(norm_kw in norm_node_name or norm_kw in norm_node_op 
                           for norm_kw in normalized_keywords)
            
            if is_match:
                if node.name not in seen:
                    seen.add(node.name)
                    matches.append(node.name)

    # 1) Quét tầng top-level
    _check_and_collect(graph_def.node)

    # 2) Quét sâu vào các function con (nơi chứa các index 107, 133... của ông)
    for func in graph_def.library.function:
        _check_and_collect(func.node_def)

    print(f"[mixed_int8] Tìm thấy {len(matches)} nodes khớp với denylist: {matches}")
    return matches


def _convert_int8_mixed_precision(
    saved_model_dir: Path,
    calib_npy_path: Path,
    out_path: Path,
    deny_keywords: list[str],
) -> Path | None:
    """Convert saved_model -> TFLite int8, denylist các node khớp
    deny_keywords (mặc định ["MaxPool"]) để giữ float32 riêng cho chúng.
    Trả về None nếu không tìm thấy node nào khớp (không tạo file rác).
    """
    import tensorflow as tf

    if not saved_model_dir.exists():
        print(f"[mixed_int8] Không thấy saved_model tại {saved_model_dir} "
              f"-> bỏ qua (cần keep_intermediate=True hoặc chưa dọn dẹp).")
        return None

    deny_nodes = _find_deny_nodes(saved_model_dir, deny_keywords)
    if not deny_nodes:
        print(f"[mixed_int8] Không tìm thấy node nào khớp {deny_keywords} "
              f"trong saved_model -> bỏ qua bước mixed precision.")
        return None

    print(f"[mixed_int8] Denylist {len(deny_nodes)} node (giữ float32):")
    for n in deny_nodes:
        print(f"    - {n}")

    calib_data = np.load(calib_npy_path)  # (N,H,W,3) float32, đã normalize

    def rep_ds():
        for x in calib_data:
            yield [x[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback float32 cho node bị denylist
    ]

    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        denylisted_nodes=deny_nodes
    )
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter,
        debug_dataset=rep_ds,
        debug_options=debug_options,
    )
    debugger.run()
    quantized_model = debugger.get_nondebug_quantized_model()

    out_path.write_bytes(quantized_model)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[mixed_int8] -> {out_path.name}  {size_mb:.2f} MB")
    return out_path


def dump_layer_error_stats(
    saved_model_dir: str,
    calib_npy_path: str,
    out_json: str | None = None,
) -> "object":
    """Chạy QuantizationDebugger.layer_statistics_dump() để xem layer nào
    lỗi lượng tử hóa lớn nhất — dùng để XÁC NHẬN đúng MaxPool là thủ phạm
    (hay thật ra là ECA's Conv1D/Sigmoid) TRƯỚC khi denylist. Gọi hàm này
    độc lập, không nằm trong convert() chính.
    """
    import tensorflow as tf

    calib_data = np.load(calib_npy_path)

    def rep_ds():
        for x in calib_data:
            yield [x[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter, debug_dataset=rep_ds,
    )
    debugger.run()
    df = debugger.layer_statistics_dump()
    print(df.sort_values("rmse/scale", ascending=False).head(20))
    if out_json:
        df.to_json(out_json, orient="records", indent=2)
        print(f"[debug] Đã ghi bảng lỗi layer -> {out_json}")
    return df


def convert(
    onnx_path: str,
    out_dir: str,
    input_size: int = 224,
    mode: str = "all",  # "fp32" | "fp16" | "int8" | "all"
    calib_dir: str | None = None,
    n_calib: float = 0.10,
    # ^ LƯU Ý: n_calib là TỈ LỆ (fraction) ảnh mỗi class để calibrate.
    calib_split_name: str | None = "test_set",
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    keep_intermediate: bool = False,
    mixed_precision_int8: bool = False,
    mixed_precision_keywords: list[str] | None = None,
) -> dict[str, Path]:
    _check_deps()
    import onnx2tf

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {onnx_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = out_dir / "saved_model"

    need_int8 = mode in ("int8", "all")
    # Mixed-precision cần đọc lại saved_model sau khi onnx2tf xong -> ép giữ
    # intermediate cho tới khi xử lý xong bước đó, dọn dẹp ở cuối như cũ.
    need_saved_model_after = need_int8 and mixed_precision_int8

    print(f"[1/2] Convert ONNX -> TF SavedModel + TFLite (onnx2tf): {onnx_path.name}")

    # onnx2tf phiên bản cũ cần 1 file cache "legacy" nằm ở cwd, tự tạo dummy nếu thiếu.
    legacy_cache_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    legacy_cache_path = Path.cwd() / legacy_cache_name
    legacy_cache_created = False
    if not legacy_cache_path.exists():
        dummy_legacy = np.random.rand(20, 128, 128, 3).astype(np.float32)
        np.save(legacy_cache_path, dummy_legacy)
        legacy_cache_created = True

    test_data_path = out_dir / "_dummy_test_data.npy"
    dummy = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
    np.save(test_data_path, dummy)

    kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        test_data_nhwc_path=str(test_data_path),
    )

    npy_calib_path = None
    if need_int8:
        if calib_dir:
            input_name = _get_onnx_input_name(onnx_path)
            npy_calib_path = _build_calibration_npy(
                calib_dir, input_size, n_calib, out_dir, mean=mean, std=std,
                split_name=calib_split_name,
            )
            # Ảnh trong npy đã normalize (mean/std) sẵn -> báo onnx2tf mean=0, std=1
            # để nó KHÔNG normalize thêm lần nữa.
            kwargs["custom_input_op_name_np_data_path"] = [
                [input_name, str(npy_calib_path), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            ]
            kwargs["output_integer_quantized_tflite"] = True
        else:
            if mixed_precision_int8:
                print("[CẢNH BÁO] mixed_precision_int8=True nhưng không có --calib-dir "
                      "-> cần ảnh calib thật để chạy QuantizationDebugger, tắt mixed precision.")
                mixed_precision_int8 = False
                need_saved_model_after = False
            print("[INFO] Không có --calib-dir -> dùng Dynamic Range Quantization "
                  "(int8 weight-only, không cần ảnh calib).")
            kwargs["output_dynamic_range_quantized_tflite"] = True

    # ── Param replacement: ưu tiên file TĨNH cạnh script, nếu có dùng luôn ──
    used_static_param_file = False
    if PARAM_REPLACEMENT_FILE.exists():
        print(f"[param_replacement] Dùng file tĩnh có sẵn: {PARAM_REPLACEMENT_FILE}")
        kwargs["param_replacement_file"] = str(PARAM_REPLACEMENT_FILE)
        used_static_param_file = True

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
        if used_static_param_file:
            # Đã dùng file tĩnh rồi mà vẫn lỗi -> không retry mù, báo luôn.
            print(f"[LỖI] Convert thất bại dù đã dùng param_replacement.json tĩnh: {e}")
            raise
        if auto_json_path.exists():
            print(f"\n[RETRY] Convert lỗi ({e.__class__.__name__}). onnx2tf đã tự sinh "
                  f"file sửa layout tại:\n         {auto_json_path}\n"
                  f"         -> Thử convert lại với param_replacement_file...")
            kwargs["param_replacement_file"] = str(auto_json_path)
            try:
                _do_convert(**kwargs)
                print("[RETRY] Convert lại thành công với auto JSON! "
                      f"Khuyên: copy nội dung {auto_json_path} thành "
                      f"{PARAM_REPLACEMENT_FILE.name} cạnh script để lần sau khỏi retry.")
            except Exception as e2:
                print(f"[LỖI] Convert lại với auto JSON vẫn thất bại: {e2}")
                raise
        else:
            print(f"[THÔNG BÁO] Không tìm thấy file JSON tự động tại: {auto_json_path}")
            raise

    print("[2/2] Rename + copy .tflite cần dùng ra out_dir gốc")
    produced = list(saved_model_dir.glob("*.tflite"))
    if not produced:
        print("[CẢNH BÁO] Không thấy file .tflite nào được sinh ra — kiểm tra log onnx2tf ở trên.")
        return {}

    wanted_keys = ("fp32", "fp16", "int8") if mode == "all" else (mode,)

    result: dict[str, Path] = {}
    for key in wanted_keys:
        match = next((f for f in produced if any(k in f.name for k in SOURCE_KEYWORDS[key])), None)
        if match is None:
            continue
        dst = out_dir / FINAL_NAMES[key]
        shutil.copy2(match, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  -> {dst.name:<28} {size_mb:.2f} MB")
        result[key] = dst

    if not result:
        print(f"[CẢNH BÁO] Không có file nào khớp --mode, kiểm tra lại tên file trong {saved_model_dir}")

    # ── Mixed-precision int8: chạy TRƯỚC khi dọn saved_model ──────────
    if need_int8 and mixed_precision_int8 and npy_calib_path is not None:
        keywords = mixed_precision_keywords or ["MaxPool2D"]
        print(f"[mixed_int8] Đang xuất bản int8 denylist keyword {keywords} ...")
        mixed_path = _convert_int8_mixed_precision(
            saved_model_dir=saved_model_dir,
            calib_npy_path=npy_calib_path,
            out_path=out_dir / MIXED_INT8_NAME,
            deny_keywords=keywords,
        )
        if mixed_path is not None:
            result["int8_mixed"] = mixed_path

    # ── Dọn rác trung gian ──────────────────────────────────────────
    if not keep_intermediate:
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir, ignore_errors=True)
        for p in (test_data_path, npy_calib_path):
            if p is not None and Path(p).exists():
                Path(p).unlink(missing_ok=True)
        if legacy_cache_created and legacy_cache_path.exists():
            legacy_cache_path.unlink(missing_ok=True)
        print("[dọn dẹp] Đã xóa saved_model/ + file tạm — out_dir chỉ còn .tflite cuối cùng.")

    print(f"\nXong. Các file .tflite nằm trong '{out_dir}'.")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Eval accuracy ONNX + TFLite trên test set thật (chạy trong venv-tflite)
# ═══════════════════════════════════════════════════════════════════════

def _stratified_subsample(
    paths: list[Path],
    labels: list[int],
    n_classes: int,
    max_samples: int | None = None,
    sample_fraction: float | None = None,
    min_per_class: int = 5,
    seed: int = 42,
) -> tuple[list[Path], list[int]]:
    """Lấy mẫu RẢI ĐỀU theo từng class (stratified) để eval nhanh.

    - sample_fraction: vd 0.1 -> lấy 10% ảnh mỗi class.
    - max_samples: giới hạn TỔNG số ảnh cuối cùng (vd 1000) — chia đều
      cho n_classes, mỗi class lấy tối đa max_samples // n_classes.
    - Nếu cả 2 đều None -> trả nguyên, không subsample.
    - min_per_class: đảm bảo mỗi class có ít nhất N ảnh.
    """
    if max_samples is None and sample_fraction is None:
        return paths, labels

    rng = np.random.default_rng(seed)

    by_class: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(idx)

    per_class_cap = None
    if max_samples is not None:
        per_class_cap = max(min_per_class, max_samples // max(n_classes, 1))

    out_paths: list[Path] = []
    out_labels: list[int] = []
    for lbl, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        if sample_fraction is not None:
            k = max(min_per_class, int(round(len(idxs) * sample_fraction)))
        else:
            k = per_class_cap
        k = min(k, len(idxs))
        for i in idxs[:k]:
            out_paths.append(paths[i])
            out_labels.append(labels[i])

    return out_paths, out_labels


def _load_test_set(
    test_dir: str,
    input_size: int,
    class_names: list[str] | None = None,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    split_name: str | None = "test_set",
    max_samples: int | None = None,
    sample_fraction: float | None = None,
    seed: int = 42,
):
    """Chỉ QUÉT danh sách (path, label) — KHÔNG load ảnh vào RAM ở đây.
    Ảnh được đọc từng cái một lúc infer (streaming) trong
    _eval_onnx_backend / _eval_tflite_backend.

    Tự nhận diện flat ImageFolder hay CCMT 2 tầng
    (root/<Group>/<split_name>/<class>/*.jpg).

    Nếu class_names được truyền vào (đúng thứ tự lúc train) thì dùng luôn,
    KHÔNG tự sort lại — để đảm bảo khớp index model đã học. Nếu để None thì
    tự sort theo alphabet.

    max_samples / sample_fraction: xem _stratified_subsample. Mặc định cả
    2 đều None -> dùng TOÀN BỘ dataset (streaming nên không OOM, nhưng sẽ
    CHẬM với dataset lớn — subsample để chạy nhanh hơn nhiều).

    Trả về (list[Path], list[int] labels, list[str] classes).
    """
    test_dir = Path(test_dir)
    raw, scanned_classes = _scan_dataset(test_dir, split_name=split_name)

    classes = class_names or scanned_classes
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    paths: list[Path] = []
    labels: list[int] = []
    skipped_unknown = 0
    for p, cname in raw:
        if cname not in cls_to_idx:
            skipped_unknown += 1
            continue
        paths.append(p)
        labels.append(cls_to_idx[cname])

    if skipped_unknown:
        print(f"  [!] Bỏ qua {skipped_unknown} ảnh có class không khớp với "
              f"class_names truyền vào — kiểm tra lại danh sách class nếu số này lớn.")

    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào (khớp class) trong {test_dir}")

    if max_samples is not None or sample_fraction is not None:
        n_before = len(paths)
        paths, labels = _stratified_subsample(
            paths, labels, n_classes=len(classes),
            max_samples=max_samples, sample_fraction=sample_fraction, seed=seed,
        )
        tag = f"fraction={sample_fraction}" if sample_fraction is not None else f"max_samples={max_samples}"
        print(f"  [sample] Subsample stratified: {n_before} -> {len(paths)} ảnh ({tag})")

    return paths, labels, classes


def _load_one_image(
    path: Path,
    input_size: int,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
) -> np.ndarray:
    """Đọc + resize + normalize 1 ảnh, trả về (1, H, W, 3) float32 NHWC."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((input_size, input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean_arr) / std_arr
    return arr[None, ...]  # (1, H, W, 3)


def _eval_onnx_backend(
    onnx_path: Path,
    paths: list[Path],
    labels: list[int],
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    log_every: int = 2000,
) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    correct, total = 0, 0
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        img_nhwc = _load_one_image(p, input_size, mean_arr, std_arr)
        img_nchw = np.transpose(img_nhwc, (0, 3, 1, 2)).astype(np.float32)
        logits = sess.run(None, {input_name: img_nchw})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        correct += int(pred == labels[i])
        total += 1
        if log_every and total % log_every == 0:
            print(f"       ...[onnx] {total}/{len(paths)} ảnh")
    elapsed = time.perf_counter() - t0
    return {"acc": correct / total, "ms_per_img": elapsed / max(total, 1) * 1000}


def _eval_tflite_backend(
    tflite_path: Path,
    paths: list[Path],
    labels: list[int],
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    log_every: int = 2000,
) -> dict:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    is_quant_in  = input_details["dtype"] in (np.int8, np.uint8)
    is_quant_out = output_details["dtype"] in (np.int8, np.uint8)

    correct, total = 0, 0
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        x = _load_one_image(p, input_size, mean_arr, std_arr)
        if is_quant_in:
            scale, zero_point = input_details["quantization"]
            x = (x / scale + zero_point).round().astype(input_details["dtype"])
        else:
            x = x.astype(np.float32)

        interpreter.resize_tensor_input(input_details["index"], (1, *x.shape[1:]))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if is_quant_out:
            scale, zero_point = output_details["quantization"]
            out = (out.astype(np.float32) - zero_point) * scale
        pred = int(np.argmax(out, axis=1)[0])
        correct += int(pred == labels[i])
        total += 1
        if log_every and total % log_every == 0:
            print(f"       ...[{tflite_path.stem}] {total}/{len(paths)} ảnh")
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
    split_name: str | None = "test_set",
    max_samples: int | None = 1000,
    sample_fraction: float | None = None,
    seed: int = 42,
) -> dict:
    """max_samples mặc định 1000 (stratified đều theo class) để eval nhanh —
    truyền max_samples=None + sample_fraction=... nếu muốn lấy theo tỉ lệ %
    mỗi class thay vì tổng số cố định. Truyền cả 2 = None để eval TOÀN BỘ
    dataset (chậm nhưng không OOM vì đã streaming)."""
    print(f"[eval] Đang quét test set từ {test_dir} (mean={mean}, std={std}) ...")
    paths, labels, classes = _load_test_set(
        test_dir, input_size, class_names, mean=mean, std=std, split_name=split_name,
        max_samples=max_samples, sample_fraction=sample_fraction, seed=seed,
    )
    print(f"[eval] {len(labels)} ảnh, {len(classes)} class.")

    results: dict[str, dict] = {}

    onnx_path = Path(onnx_path)
    if onnx_path.exists():
        print("[eval] Đang test ONNX Runtime...")
        results["onnx"] = _eval_onnx_backend(onnx_path, paths, labels, input_size, mean, std)
        print(f"       acc={results['onnx']['acc']*100:.2f}%  {results['onnx']['ms_per_img']:.2f} ms/ảnh")

    # 3 file cố định + bản mixed-precision int8 nếu có.
    eval_targets = dict(FINAL_NAMES)
    eval_targets["int8_mixed"] = MIXED_INT8_NAME

    for key, fname in eval_targets.items():
        tflite_path = Path(tflite_dir) / fname
        if not tflite_path.exists():
            continue
        print(f"[eval] Đang test TFLite [{key}]...")
        try:
            results[key] = _eval_tflite_backend(tflite_path, paths, labels, input_size, mean, std)
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
    """CLI optional — chỉ dùng khi muốn chạy độc lập file này ngoài
    re_export.py. Bình thường re_export.py gọi thẳng convert()/
    evaluate_backends() bằng Python, không cần CLI này."""
    ap = argparse.ArgumentParser(description="Convert ONNX -> TFLite (fp32/fp16/int8) + eval accuracy")
    ap.add_argument("--onnx", required=True, help="Path tới best_deploy.onnx")
    ap.add_argument("--out", default="weights/tflite", help="Thư mục xuất .tflite")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--mode", choices=["fp32", "fp16", "int8", "all"], default="all")
    ap.add_argument("--calib-dir", default=None,
                     help="Thư mục ảnh thật để calibrate int8 (bắt buộc nếu mode=int8/all). "
                          "Có thể trỏ tới root ImageFolder phẳng HOẶC root CCMT 2 tầng "
                          "(vd 'CCMT Dataset-Augmented') — tự nhận diện cấu trúc.")
    ap.add_argument("--n-calib", type=float, default=0.20,
                     help="Tỉ lệ ảnh MỖI class dùng để calibrate int8 (mặc định 0.10 = 10%%). "
                          "Tăng lên 0.2 nếu accuracy int8 chưa ổn.")
    ap.add_argument("--calib-split-name", default="test_set",
                     help="Chỉ dùng khi --calib-dir là CCMT 2 tầng: tên split để lấy ảnh calib "
                          "(vd 'test_set' hoặc 'train_set'). Bỏ qua nếu --calib-dir là ImageFolder phẳng.")
    ap.add_argument("--test-dir", default=None,
                     help="Thư mục test THẬT để eval accuracy sau convert. Có thể trỏ tới root "
                          "ImageFolder phẳng (class_name/*.jpg) HOẶC root CCMT 2 tầng "
                          "(Group/split/class/*.jpg) — tự nhận diện.")
    ap.add_argument("--test-split-name", default="test_set",
                     help="Chỉ dùng khi --test-dir là CCMT 2 tầng: tên split chứa ảnh test "
                          "(vd 'test_set'). Bỏ qua nếu --test-dir là ImageFolder phẳng.")
    ap.add_argument("--class-names", default=None,
                     help="Danh sách class cách nhau bởi dấu phẩy, PHẢI đúng thứ tự index lúc train")
    ap.add_argument("--eval-out", default=None,
                     help="Path json ghi kết quả eval (mặc định: <out>/backend_eval_results.json)")
    ap.add_argument("--sample-fraction", type=float, default=0.10,
                     help="Lấy X%% ảnh MỖI class để eval accuracy sau convert (mặc định 0.10 = 10%%).")
    ap.add_argument("--mean", default=None)
    ap.add_argument("--std", default=None)
    ap.add_argument("--keep-intermediate", action="store_true",
                     help="Giữ lại saved_model/ + file tạm (debug), mặc định luôn dọn sạch")
    ap.add_argument("--mixed-int8", action="store_true",
                     help="Xuất thêm bản int8 denylist các node khớp --mixed-int8-keywords "
                          "(giữ float32 riêng cho chúng), cần --calib-dir.")
    ap.add_argument("--mixed-int8-keywords", default="MaxPool2D",
                     help="Từ khóa (cách nhau bởi dấu phẩy) để tìm node denylist, "
                          "khớp cả tên node lẫn op type. Mặc định 'MaxPool2D'.")
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
        calib_split_name=args.calib_split_name,
        mean=mean,
        std=std,
        keep_intermediate=args.keep_intermediate,
        mixed_precision_int8=args.mixed_int8,
        mixed_precision_keywords=[k.strip() for k in args.mixed_int8_keywords.split(",") if k.strip()],
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
            split_name=args.test_split_name,
            max_samples=None,
            sample_fraction=args.sample_fraction,
        )


if __name__ == "__main__":
    _cli()