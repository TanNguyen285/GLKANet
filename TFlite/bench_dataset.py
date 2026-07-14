from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from PIL import Image
except ImportError:
    print("❌ Cần cài Pillow: pip install pillow --break-system-packages")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("❌ Cần cài pyyaml: pip install pyyaml --break-system-packages")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
#  ⚙️  CONFIG — SỬA Ở ĐÂY, RỒI CHẠY THẲNG: python benchmark_dataset.py
#     Không cần truyền CLI flag nào nữa (CLI vẫn còn nhưng optional, để
#     override nếu muốn — mặc định dùng luôn giá trị trong CONFIG).
# ══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Đường dẫn dataset.yaml (path/train/test theo format CCMT)
    "dataset_yaml": r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\dataset.yaml",

    # Model cần benchmark — để None nếu không muốn chạy loại đó
    "onnx": r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_augument\weights\best_deploy.onnx",
    "tflite": None,
    "tflite_dir": r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_augument\weights\tflite",

    "input_size": 224,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "onnx_provider": "CPU",       # "CPU" | "CUDA" | "TensorRT"
    "limit": None,                # chỉ chạy N ảnh đầu (test nhanh), None = chạy hết
    "num_threads_list": [2, 4, 8],  # để [] hoặc None -> chỉ chạy 1 lần config mặc định

    "out_dir": "./bench_results",

    # % ảnh MỖI class lấy từ test_set để benchmark accuracy/latency
    "auto_sample_fraction": 0.10,
    "auto_sample_min_per_class": 5,
    "auto_sample_seed": 42,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp"}
WARMUP_RUNS = 100          # cố định — mọi model đều warmup đúng 100 lần
PAUSE_BETWEEN_MODELS_S = 3.0  # cố định — nghỉ đúng 3 giây giữa 2 model


# ══════════════════════════════════════════════════════════════════════════
#  Đọc dataset.yaml + quét test_set thật (cấu trúc CCMT 2 tầng: Group/split/class)
# ══════════════════════════════════════════════════════════════════════════

def _stratified_subsample_paths(
    raw: list[tuple[str, str]],
    fraction: float,
    min_per_class: int = 5,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Lấy rải đều `fraction` ảnh mỗi class (vd 0.10 = 10%), đảm bảo
    min_per_class ảnh tối thiểu cho class hiếm. raw = [(path, class_name), ...]."""
    rng = np.random.default_rng(seed)

    by_class: dict[str, list[str]] = {}
    for path, cname in raw:
        by_class.setdefault(cname, []).append(path)

    out: list[tuple[str, str]] = []
    for cname, paths in by_class.items():
        paths = list(paths)
        idx = np.arange(len(paths))
        rng.shuffle(idx)
        k = max(min_per_class, int(round(len(paths) * fraction)))
        k = min(k, len(paths))
        for i in idx[:k]:
            out.append((paths[i], cname))
    return out


def load_test_set(
    dataset_yaml: str,
    sample_fraction: float | None,
    min_per_class: int,
    seed: int,
) -> tuple[list[str], list[int], list[str]]:
    """Đọc dataset.yaml dạng:
        path: <root>
        train: train_set
        test: test_set
    quét <root>/<Group>/<test>/<class_name>/*.ảnh (cấu trúc CCMT 2 tầng),
    rồi tự động lấy sample_fraction ảnh mỗi class (nếu không None).

    class_names có dạng "<Group>_<class>" (vd "Cashew_anthracnose"), khớp
    format lúc train. Label = index trong sorted(class_names) toàn cục.
    """
    cfg = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    split_name = cfg["test"]  # vd "test_set"

    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy root dataset: {root}")

    group_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not group_dirs:
        raise FileNotFoundError(f"Không có group folder nào (Cashew/Cassava/...) trong: {root}")

    # normalize: bỏ số cuối tên class, vd "anthracnose1" -> "anthracnose"
    def _norm(name: str) -> str:
        return re.sub(r'\d+$', '', name).strip()

    raw: list[tuple[str, str]] = []  # (path, full_class_name)
    class_names_set: set[str] = set()
    found_any_split = False

    for group_dir in group_dirs:
        split_dir = group_dir / split_name
        if not split_dir.exists():
            continue
        found_any_split = True
        group_name = group_dir.name
        for cd in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            cls = _norm(cd.name)
            full_name = f"{group_name}_{cls}"
            class_names_set.add(full_name)
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix in IMG_EXTS:
                    raw.append((str(f), full_name))

    if not found_any_split:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục '{split_name}/' trong bất kỳ group nào tại: {root}\n"
            f"Group tìm thấy: {[d.name for d in group_dirs]}"
        )
    if not raw:
        raise FileNotFoundError(f"Không có ảnh nào trong split '{split_name}' tại: {root}")

    class_names = sorted(class_names_set)

    # ── Auto-sample X% mỗi class ──
    if sample_fraction is not None:
        n_before = len(raw)
        raw = _stratified_subsample_paths(
            raw, fraction=sample_fraction,
            min_per_class=min_per_class, seed=seed,
        )
        print(f"  [auto-sample] {n_before} -> {len(raw)} ảnh "
              f"(lấy {sample_fraction*100:.0f}% mỗi class, tối thiểu "
              f"{min_per_class} ảnh/class, seed={seed})")

    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    image_paths = [p for p, _ in raw]
    labels = [cls_to_idx[c] for _, c in raw]

    return image_paths, labels, class_names


def preprocess_image(path: str, input_size: int, mean, std) -> np.ndarray | None:
    """Trả về mảng NCHW float32 (1, 3, H, W) đã normalize, hoặc None nếu ảnh lỗi."""
    try:
        img = Image.open(path)
        img.load()  # ép đọc hết data ngay để bắt lỗi broken stream sớm, tránh crash giữa chừng
        img = img.convert("RGB").resize((input_size, input_size), Image.BILINEAR)
    except Exception:
        return None
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)                              # CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)      # NCHW
    return arr


# ══════════════════════════════════════════════════════════════════════════
#  Thống kê (giữ nguyên công thức mean/std/CI95 chuẩn, t-distribution cho n nhỏ)
# ══════════════════════════════════════════════════════════════════════════

def _t_critical_95(n: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042,
        50: 2.009, 100: 1.984,
    }
    df = n - 1
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    closest = min(table.keys(), key=lambda k: abs(k - df))
    return table[closest] if df < 120 else 1.96


def compute_stats(times_ms: list[float]) -> dict:
    n = len(times_ms)
    ts = sorted(times_ms)
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    tcrit = _t_critical_95(n) if n > 1 else float("nan")
    ci95 = tcrit * sem if n > 1 else 0.0

    def pct(p):
        idx = min(int(len(ts) * p), len(ts) - 1)
        return ts[idx]

    return {
        "n_images": n,
        "latency_mean_ms": round(mean, 4),
        "latency_std_ms": round(std, 4),
        "cv_percent": round((std / mean) * 100, 2) if mean else 0.0,
        "ci95_ms": round(ci95, 4),
        "latency_min_ms": round(min(times_ms), 4),
        "latency_max_ms": round(max(times_ms), 4),
        "latency_median_ms": round(statistics.median(times_ms), 4),
        "latency_p50_ms": round(pct(0.50), 4),
        "latency_p90_ms": round(pct(0.90), 4),
        "latency_p95_ms": round(pct(0.95), 4),
        "latency_p99_ms": round(pct(0.99), 4),
        "throughput_fps": round(1000.0 / mean, 3) if mean else 0.0,
        "total_time_s": round(sum(times_ms) / 1000.0, 3),
    }


class _MemSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss_mb = 0.0
        self.baseline_rss_mb = 0.0

    def start(self):
        if not HAS_PSUTIL:
            return
        proc = psutil.Process(os.getpid())
        self.baseline_rss_mb = proc.memory_info().rss / 1e6
        self.peak_rss_mb = self.baseline_rss_mb

        def _loop():
            while not self._stop.is_set():
                try:
                    rss = proc.memory_info().rss / 1e6
                    if rss > self.peak_rss_mb:
                        self.peak_rss_mb = rss
                except Exception:
                    pass
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not HAS_PSUTIL:
            return {"memory_available": False}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        return {
            "memory_available": True,
            "baseline_rss_mb": round(self.baseline_rss_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "delta_rss_mb": round(self.peak_rss_mb - self.baseline_rss_mb, 2),
        }


# ══════════════════════════════════════════════════════════════════════════
#  Benchmark ONNX trên (đã auto-sample) test_set
# ══════════════════════════════════════════════════════════════════════════

def benchmark_onnx_dataset(
    model_path: str,
    image_paths: list[str],
    labels: list[int],
    input_size: int,
    mean, std,
    provider_name: str = "CPU",
    num_threads: int | None = None,
) -> dict:
    import onnxruntime as ort

    provider_map = {"CPU": "CPUExecutionProvider", "CUDA": "CUDAExecutionProvider", "TensorRT": "TensorrtExecutionProvider"}
    requested = provider_map.get(provider_name, "CPUExecutionProvider")
    available = ort.get_available_providers()
    if requested not in available:
        return {"error": f"{requested} không có sẵn. Available: {available}"}

    sess_options = ort.SessionOptions()
    if num_threads is not None:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1

    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=[requested])
    except Exception as ex:
        return {"error": f"Không tạo được session: {ex}"}

    input_name = sess.get_inputs()[0].name
    actual_provider = sess.get_providers()[0]

    print(f"  [preprocess] Đang decode + resize {len(image_paths)} ảnh...")
    t_pre0 = time.perf_counter()
    tensors, valid_labels, skipped = [], [], 0
    for p, lb in zip(image_paths, labels):
        arr = preprocess_image(p, input_size, mean, std)
        if arr is None:
            skipped += 1
            continue
        tensors.append(arr)
        valid_labels.append(lb)
    preprocess_time_s = round(time.perf_counter() - t_pre0, 3)
    print(f"  [preprocess] Xong trong {preprocess_time_s}s ({skipped} ảnh lỗi bị bỏ qua)")

    if not tensors:
        return {"error": "Không có ảnh hợp lệ nào sau preprocess."}

    print(f"  [warmup] Chạy warmup {WARMUP_RUNS} lần...")
    for i in range(WARMUP_RUNS):
        sess.run(None, {input_name: tensors[i % len(tensors)]})

    print(f"  [benchmark] Chạy inference trên {len(tensors)} ảnh...")
    mem = _MemSampler()
    mem.start()
    times_ms = []
    correct = 0
    for arr, lb in zip(tensors, valid_labels):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: arr})[0]
        times_ms.append((time.perf_counter() - t0) * 1000)
        pred = int(np.argmax(out, axis=1)[0])
        if pred == lb:
            correct += 1
    mem_result = mem.stop()

    result = compute_stats(times_ms)
    result["backend"] = "onnx"
    result["provider_requested"] = requested
    result["provider_used"] = actual_provider
    result["warmup_runs"] = WARMUP_RUNS
    result["skipped_images"] = skipped
    result["preprocess_time_s"] = preprocess_time_s
    result["accuracy"] = round(correct / len(valid_labels), 4)
    result["correct"] = correct
    result["num_threads_requested"] = num_threads if num_threads is not None else "default"
    result.update(mem_result)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Benchmark TFLite trên (đã auto-sample) test_set
# ══════════════════════════════════════════════════════════════════════════

def _get_tflite_interpreter_cls():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter, "tensorflow"


def benchmark_tflite_dataset(
    model_path: str,
    image_paths: list[str],
    labels: list[int],
    input_size: int,
    mean, std,
    num_threads: int | None = None,
) -> dict:
    Interpreter, backend_name = _get_tflite_interpreter_cls()

    try:
        kwargs = {"model_path": model_path}
        if num_threads is not None:
            kwargs["num_threads"] = num_threads
        interp = Interpreter(**kwargs)
        interp.allocate_tensors()
    except Exception as ex:
        return {"error": f"Không tạo được interpreter: {ex}"}

    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    in_index, out_index = in_details["index"], out_details["index"]
    in_dtype = in_details["dtype"]
    tflite_input_is_nhwc = in_details["shape"][-1] == 3

    print(f"  [preprocess] Đang decode + resize {len(image_paths)} ảnh...")
    t_pre0 = time.perf_counter()
    tensors, valid_labels, skipped = [], [], 0
    for p, lb in zip(image_paths, labels):
        arr = preprocess_image(p, input_size, mean, std)  # NCHW float32
        if arr is None:
            skipped += 1
            continue
        if tflite_input_is_nhwc:
            arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        if in_dtype != np.float32:
            scale, zero_point = in_details.get("quantization", (0.0, 0))
            if scale == 0:
                skipped += 1
                continue
            arr = (arr / scale + zero_point).round().astype(in_dtype)
        else:
            arr = arr.astype(np.float32)
        tensors.append(arr)
        valid_labels.append(lb)
    preprocess_time_s = round(time.perf_counter() - t_pre0, 3)
    print(f"  [preprocess] Xong trong {preprocess_time_s}s ({skipped} ảnh lỗi bị bỏ qua)")

    if not tensors:
        return {"error": "Không có ảnh hợp lệ nào sau preprocess."}

    def _run_once(arr):
        interp.set_tensor(in_index, arr)
        interp.invoke()
        return interp.get_tensor(out_index)

    print(f"  [warmup] Chạy warmup {WARMUP_RUNS} lần...")
    for i in range(WARMUP_RUNS):
        _run_once(tensors[i % len(tensors)])

    print(f"  [benchmark] Chạy inference trên {len(tensors)} ảnh...")
    mem = _MemSampler()
    mem.start()
    times_ms = []
    correct = 0
    out_scale, out_zero_point = out_details.get("quantization", (0.0, 0))
    for arr, lb in zip(tensors, valid_labels):
        t0 = time.perf_counter()
        out = _run_once(arr)
        times_ms.append((time.perf_counter() - t0) * 1000)
        if out_details["dtype"] != np.float32 and out_scale != 0:
            out = (out.astype(np.float32) - out_zero_point) * out_scale
        pred = int(np.argmax(out, axis=1)[0])
        if pred == lb:
            correct += 1
    mem_result = mem.stop()

    result = compute_stats(times_ms)
    result["backend"] = "tflite"
    result["tflite_backend_lib"] = backend_name
    result["input_layout"] = "NHWC" if tflite_input_is_nhwc else "NCHW"
    result["input_dtype"] = str(in_dtype)
    result["warmup_runs"] = WARMUP_RUNS
    result["skipped_images"] = skipped
    result["preprocess_time_s"] = preprocess_time_s
    result["accuracy"] = round(correct / len(valid_labels), 4)
    result["correct"] = correct
    result["num_threads_requested"] = num_threads if num_threads is not None else "default"
    result.update(mem_result)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  In kết quả + main
# ══════════════════════════════════════════════════════════════════════════

def print_result(label: str, r: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  KẾT QUẢ: {label}")
    print(f"{'='*70}")
    if "error" in r:
        print(f"  ❌ LỖI: {r['error']}")
        return
    print(f"  Số ảnh chạy         : {r['n_images']} (bỏ qua {r['skipped_images']} ảnh lỗi)")
    print(f"  Số thread yêu cầu   : {r.get('num_threads_requested', 'default')}")
    print(f"  Tổng thời gian infer: {r['total_time_s']} giây (KHÔNG tính preprocess/warmup)")
    print(f"  Thời gian preprocess: {r['preprocess_time_s']} giây (decode + resize toàn bộ ảnh)")
    print(f"  Latency mean/median : {r['latency_mean_ms']} / {r['latency_median_ms']} ms (CV {r['cv_percent']}%)")
    print(f"  Latency P50/P90/P99 : {r['latency_p50_ms']} / {r['latency_p90_ms']} / {r['latency_p99_ms']} ms")
    print(f"  Throughput          : {r['throughput_fps']} ảnh/giây (FPS)")
    print(f"  Accuracy            : {r['accuracy']*100:.2f}% ({r['correct']}/{r['n_images']})")
    if r.get("memory_available"):
        print(f"  Peak RSS memory     : {r['peak_rss_mb']} MB (delta {r['delta_rss_mb']} MB)")
    if r.get("cv_percent", 0) > 30:
        print(f"  ⚠️ CV cao ({r['cv_percent']}%) — máy có thể đang bận tiến trình nền khác lúc đo")


def run(cfg: dict) -> None:
    """Chạy full benchmark theo CONFIG. Đây là hàm chính — gọi trực tiếp,
    không cần CLI."""
    if not cfg["onnx"] and not cfg["tflite"] and not cfg["tflite_dir"]:
        print("❌ CONFIG cần ít nhất 'onnx' hoặc 'tflite'/'tflite_dir'.")
        sys.exit(1)

    mean = tuple(cfg["mean"])
    std = tuple(cfg["std"])

    print(f"=== Đọc dataset từ: {cfg['dataset_yaml']} ===")
    image_paths, labels, class_names = load_test_set(
        cfg["dataset_yaml"],
        sample_fraction=cfg["auto_sample_fraction"],
        min_per_class=cfg["auto_sample_min_per_class"],
        seed=cfg["auto_sample_seed"],
    )
    print(f"  Tìm thấy {len(class_names)} class, tổng {len(image_paths)} ảnh sau auto-sample.")

    if cfg["limit"]:
        image_paths = image_paths[:cfg["limit"]]
        labels = labels[:cfg["limit"]]
        print(f"  limit={cfg['limit']} → chỉ chạy {len(image_paths)} ảnh đầu tiên.")

    thread_list = cfg["num_threads_list"] or [None]
    print(f"  Cấu hình thread sẽ chạy: {thread_list}")

    all_results: dict[str, dict] = {}

    for cfg_idx, nt in enumerate(thread_list):
        nt_label = f"{nt}threads" if nt is not None else "default"
        print(f"\n{'='*70}\n  ═══ CẤU HÌNH: {nt_label} ═══\n{'='*70}")

        if cfg["onnx"]:
            if not Path(cfg["onnx"]).exists():
                print(f"❌ Không tìm thấy file ONNX: {cfg['onnx']}")
                sys.exit(1)
            print(f"\n{'#'*70}\n# BENCHMARK ONNX [{nt_label}]: {cfg['onnx']}\n{'#'*70}")
            t_start = time.perf_counter()
            r = benchmark_onnx_dataset(cfg["onnx"], image_paths, labels, cfg["input_size"], mean, std,
                                        cfg["onnx_provider"], num_threads=nt)
            r["wall_clock_total_s"] = round(time.perf_counter() - t_start, 3)
            key = f"onnx_{nt_label}"
            all_results[key] = {"model_path": str(Path(cfg["onnx"]).resolve()), "result": r}
            print_result(f"ONNX ({cfg['onnx_provider']}) [{nt_label}]", r)

        tflite_targets: list[Path] = []
        if cfg["tflite_dir"]:
            tflite_targets = sorted(Path(cfg["tflite_dir"]).glob("*.tflite"))
            if not tflite_targets:
                print(f"❌ Không tìm thấy file .tflite nào trong: {cfg['tflite_dir']}")
                sys.exit(1)
        elif cfg["tflite"]:
            p = Path(cfg["tflite"])
            if not p.exists():
                print(f"❌ Không tìm thấy file TFLite: {cfg['tflite']}")
                sys.exit(1)
            tflite_targets = [p]

        if cfg["onnx"] and tflite_targets:
            print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s giữa ONNX và loạt TFLite để hệ thống ổn định lại...")
            time.sleep(PAUSE_BETWEEN_MODELS_S)

        for i, tflite_path in enumerate(tflite_targets):
            if i > 0:
                print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s trước khi benchmark model tiếp theo...")
                time.sleep(PAUSE_BETWEEN_MODELS_S)
            key = f"{tflite_path.stem}_{nt_label}"
            print(f"\n{'#'*70}\n# BENCHMARK TFLITE [{key}]: {tflite_path}\n{'#'*70}")
            t_start = time.perf_counter()
            r = benchmark_tflite_dataset(str(tflite_path), image_paths, labels, cfg["input_size"], mean, std,
                                          num_threads=nt)
            r["wall_clock_total_s"] = round(time.perf_counter() - t_start, 3)
            all_results[key] = {"model_path": str(tflite_path.resolve()), "result": r}
            print_result(f"TFLite [{key}]", r)

        if cfg_idx < len(thread_list) - 1:
            print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s trước khi đổi sang cấu hình thread tiếp theo...")
            time.sleep(PAUSE_BETWEEN_MODELS_S)

    print(f"\n{'='*70}\n  SO SÁNH TỔNG HỢP (tất cả model x tất cả cấu hình thread)\n{'='*70}")
    print(f"  {'Backend':<30} {'Latency(ms)':>14} {'FPS':>10} {'Acc(%)':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*10}")
    for name, entry in all_results.items():
        r = entry["result"]
        if "error" in r:
            print(f"  {name:<30} {'--- lỗi ---':>14}")
            continue
        print(f"  {name:<30} {r['latency_mean_ms']:>14} {r['throughput_fps']:>10} {r['accuracy']*100:>9.2f}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_dataset_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset_yaml": str(Path(cfg["dataset_yaml"]).resolve()),
            "num_classes": len(class_names),
            "num_images_total": len(image_paths),
            "input_size": cfg["input_size"],
            "mean": mean, "std": std,
            "warmup_runs": WARMUP_RUNS,
            "pause_between_models_s": PAUSE_BETWEEN_MODELS_S,
            "num_threads_list": thread_list,
            "auto_sample_fraction": cfg["auto_sample_fraction"],
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu báo cáo đầy đủ: {out_path}")


def _cli_override(cfg: dict) -> dict:
    """CLI optional — chỉ dùng khi muốn override CONFIG mà không sửa code.
    Không truyền gì cả -> dùng y nguyên CONFIG ở đầu file."""
    ap = argparse.ArgumentParser(description="Benchmark ONNX/TFLite (mặc định dùng CONFIG trong file, CLI chỉ để override).")
    ap.add_argument("--dataset-yaml", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--tflite", default=None)
    ap.add_argument("--tflite-dir", default=None)
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--sample-fraction", type=float, default=None,
                     help="Override auto_sample_fraction (vd 0.2 = 20%% mỗi class)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-threads-list", default=None, help="vd '2,4,8'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset_yaml: cfg["dataset_yaml"] = args.dataset_yaml
    if args.onnx: cfg["onnx"] = args.onnx
    if args.tflite: cfg["tflite"] = args.tflite
    if args.tflite_dir: cfg["tflite_dir"] = args.tflite_dir
    if args.input_size: cfg["input_size"] = args.input_size
    if args.sample_fraction is not None: cfg["auto_sample_fraction"] = args.sample_fraction
    if args.limit is not None: cfg["limit"] = args.limit
    if args.num_threads_list is not None:
        cfg["num_threads_list"] = [int(x) for x in args.num_threads_list.split(",")] if args.num_threads_list.strip() else []
    if args.out: cfg["out_dir"] = args.out
    return cfg


if __name__ == "__main__":
    cfg = _cli_override(dict(CONFIG))
    run(cfg)