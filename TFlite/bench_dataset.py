"""
bench_dataset.py — Benchmark ONNX vs TFLite chạy TRÊN TOÀN BỘ test_set thật
(không dùng dummy/random input), đọc cấu hình từ dataset.yaml (ImageFolder-style).

Đặc điểm:
    - Đọc dataset.yaml (path + test) để tự tìm thư mục test_set thật.
    - Quét toàn bộ ảnh trong test_set (cấu trúc ImageFolder: test_set/<class_name>/*.jpg).
    - Warmup CHÍNH XÁC 100 lần bằng ảnh thật (cycle lại nếu ảnh ít hơn 100) trước khi đo.
    - Nghỉ đúng 3 giây giữa lúc benchmark xong model này và trước khi bắt đầu model kế tiếp,
      để nhiệt độ CPU/GPU và bộ nhớ ổn định lại, tránh benchmark model sau bị ăn theo
      hiệu ứng cache/nhiệt của model trước.
    - Chạy batch=1 tuần tự cho từng backend để so sánh công bằng theo per-image latency.
    - Tự tính accuracy luôn (tiện có sẵn label từ tên thư mục con), không tốn thêm công.
    - Xuất báo cáo JSON đầy đủ + in bảng tổng kết ra console.

BẢN SỬA (multi-thread benchmark): thêm --num-threads-list để benchmark CÙNG 1 model
lần lượt với nhiều số CPU thread khác nhau (vd "2,4,8") — dùng để đo scaling
theo số core, đúng nhu cầu so sánh 2 vs 4 vs 8 core trên Raspberry Pi/Jetson.
Mỗi cấu hình thread sẽ tạo 1 session/interpreter RIÊNG (intra_op_num_threads cho
ONNX Runtime, num_threads cho TFLite Interpreter), có nghỉ giữa mỗi lần đo.

BẢN SỬA (trước đó): thêm --tflite-dir để tự quét TOÀN BỘ file .tflite trong 1 thư mục (vd
weights/tflite/ với best_deploy_fp32.tflite, best_deploy_fp16.tflite,
best_deploy_int8.tflite) và benchmark lần lượt từng file, có nghỉ giữa mỗi model
— dùng cho pipeline tự động gọi từ exporter.py sau khi train xong.

Chạy (trong venv-tflite vì cần cả onnxruntime lẫn tensorflow/tflite_runtime):
    pip install onnx onnxruntime numpy pillow pyyaml psutil --break-system-packages
    # (tensorflow đã có sẵn trong venv-tflite để đọc .tflite)

    # Benchmark 1 file tflite cụ thể, mặc định thread (không control):
    python bench_dataset.py ^
        --dataset-yaml "C:\\...\\dataset.yaml" ^
        --onnx "C:\\...\\best_deploy.onnx" ^
        --tflite "C:\\...\\best_deploy_fp32.tflite" ^
        --input-size 224 ^
        --num-threads-list ""

    # Benchmark cả 3 file trong 1 thư mục, lần lượt với 2/4/8 thread:
    python bench_dataset.py ^
        --dataset-yaml "C:\\...\\dataset.yaml" ^
        --onnx "C:\\...\\best_deploy.onnx" ^
        --tflite-dir "C:\\...\\weights\\tflite" ^
        --input-size 224 ^
        --num-threads-list 2,4,8

    Có thể benchmark chỉ 1 model (bỏ --onnx hoặc --tflite/--tflite-dir tương ứng).
    Dùng --limit N để chạy thử nhanh trên N ảnh đầu tiên trước khi chạy full test_set.

LƯU Ý khi đọc kết quả nhiều thread: Raspberry Pi 4/5 chỉ có 4 core vật lý — số
thread > 4 là oversubscription, thường KHÔNG nhanh hơn (có khi chậm hơn do
context-switch). Model dùng nhiều depthwise conv nhỏ (ít FLOPs/op) thường bão
hòa scaling sớm hơn model dense-conv lớn.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WARMUP_RUNS = 100          # cố định theo yêu cầu — mọi model đều warmup đúng 100 lần
PAUSE_BETWEEN_MODELS_S = 3.0  # cố định theo yêu cầu — nghỉ đúng 3 giây giữa 2 model


# ══════════════════════════════════════════════════════════════════════════
#  Đọc dataset.yaml + quét test_set thật (ImageFolder-style)
# ══════════════════════════════════════════════════════════════════════════

def load_test_set(dataset_yaml: str) -> tuple[list[str], list[int], list[str]]:
    """Đọc dataset.yaml dạng:
        path: <root>
        train: train_set
        test: test_set
    rồi quét <root>/<test>/<class_name>/*.ảnh -> trả về (image_paths, labels, class_names).

    Label được gán theo thứ tự tên thư mục con đã sort (đúng quy ước ImageFolder chuẩn:
    class index = vị trí trong danh sách tên thư mục con đã sort alphabetically).
    NẾU model của bạn train với thứ tự class khác (ví dụ đọc từ 1 file classes.txt riêng),
    accuracy tính ra ở đây có thể SAI dù tốc độ đo vẫn đúng — cần đối chiếu lại thứ tự class
    thật sự dùng lúc train nếu thấy accuracy bất thường.
    """
    cfg = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    test_dir = root / cfg["test"]

    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test_set: {test_dir}")

    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục class con nào trong: {test_dir}\n"
            f"Cấu trúc kỳ vọng: {test_dir}/<class_name>/*.jpg"
        )

    image_paths: list[str] = []
    labels: list[int] = []
    for idx, cname in enumerate(class_names):
        cdir = test_dir / cname
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                image_paths.append(str(f))
                labels.append(idx)

    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {test_dir}")

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
#  Benchmark ONNX trên full test_set
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

    # Control số thread CPU dùng cho intra-op (song song trong 1 op, vd conv
    # chia nhỏ theo channel) — đây là tham số quyết định "dùng bao nhiêu core".
    sess_options = ort.SessionOptions()
    if num_threads is not None:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1  # tắt song song giữa các op độc lập, tránh cộng dồn ngoài ý muốn

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
#  Benchmark TFLite trên full test_set
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
        # TFLite Interpreter hỗ trợ sẵn tham số num_threads trong constructor —
        # quyết định số CPU thread dùng cho các op có kernel đa luồng (XNNPACK).
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
    # TFLite convert qua onnx2tf thường chuyển input sang NHWC — tự phát hiện qua shape input.
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
            # Model đã quantize int8/uint8 — cần scale/zero_point từ quantization params.
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


def main():
    ap = argparse.ArgumentParser(description="Benchmark ONNX/TFLite trên TOÀN BỘ test_set thật (ImageFolder).")
    ap.add_argument("--dataset-yaml", required=True, help="Đường dẫn dataset.yaml (path/train/test)")
    ap.add_argument("--onnx", default=None, help="Đường dẫn model .onnx (bỏ qua nếu không muốn benchmark ONNX)")
    ap.add_argument("--tflite", default=None, help="Đường dẫn 1 model .tflite (bỏ qua nếu không muốn benchmark TFLite)")
    ap.add_argument("--tflite-dir", default=None,
                     help="Thư mục chứa nhiều file .tflite, tự quét và benchmark lần lượt "
                          "(dùng khi muốn benchmark cả fp32/fp16/int8 trong 1 lệnh). "
                          "Nếu vừa có --tflite vừa có --tflite-dir thì --tflite-dir ưu tiên.")
    ap.add_argument("--input-size", type=int, default=224, help="Kích thước ảnh vuông đưa vào model (mặc định 224)")
    ap.add_argument("--mean", default=",".join(str(x) for x in DEFAULT_MEAN), help="Mean normalize, 3 số cách nhau bởi dấu phẩy")
    ap.add_argument("--std", default=",".join(str(x) for x in DEFAULT_STD), help="Std normalize, 3 số cách nhau bởi dấu phẩy")
    ap.add_argument("--onnx-provider", default="CPU", choices=["CPU", "CUDA", "TensorRT"], help="Provider cho ONNX Runtime")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ chạy N ảnh đầu tiên (test nhanh trước khi chạy full)")
    ap.add_argument("--num-threads-list", default="2,4,8",
                     help="Danh sách số CPU thread cần benchmark, cách nhau bởi dấu phẩy "
                          "(vd '2,4,8'). Mỗi model sẽ được benchmark lại 1 lần cho MỖI số "
                          "thread trong danh sách (session/interpreter riêng cho mỗi lần), "
                          "có nghỉ giữa mỗi lần đo. Truyền chuỗi rỗng '' để chỉ chạy 1 lần "
                          "với cấu hình mặc định (không ép số thread).")
    ap.add_argument("--out", default="./bench_results", help="Thư mục lưu báo cáo JSON")
    args = ap.parse_args()

    if not args.onnx and not args.tflite and not args.tflite_dir:
        print("❌ Cần truyền ít nhất --onnx hoặc --tflite/--tflite-dir (có thể truyền cả hai để so sánh).")
        sys.exit(1)

    mean = tuple(float(x) for x in args.mean.split(","))
    std = tuple(float(x) for x in args.std.split(","))

    print(f"=== Đọc dataset từ: {args.dataset_yaml} ===")
    image_paths, labels, class_names = load_test_set(args.dataset_yaml)
    print(f"  Tìm thấy {len(class_names)} class, tổng {len(image_paths)} ảnh trong test_set.")

    if args.limit:
        image_paths = image_paths[:args.limit]
        labels = labels[:args.limit]
        print(f"  --limit={args.limit} → chỉ chạy {len(image_paths)} ảnh đầu tiên.")

    if args.num_threads_list.strip():
        thread_list: list[int | None] = [int(x) for x in args.num_threads_list.split(",")]
    else:
        thread_list = [None]
    print(f"  Cấu hình thread sẽ chạy: {thread_list}")

    all_results: dict[str, dict] = {}

    for cfg_idx, nt in enumerate(thread_list):
        nt_label = f"{nt}threads" if nt is not None else "default"
        print(f"\n{'='*70}\n  ═══ CẤU HÌNH: {nt_label} ═══\n{'='*70}")

        # ── Benchmark ONNX (nếu có) ──────────────────────────────────
        if args.onnx:
            if not Path(args.onnx).exists():
                print(f"❌ Không tìm thấy file ONNX: {args.onnx}")
                sys.exit(1)
            print(f"\n{'#'*70}\n# BENCHMARK ONNX [{nt_label}]: {args.onnx}\n{'#'*70}")
            t_start = time.perf_counter()
            r = benchmark_onnx_dataset(args.onnx, image_paths, labels, args.input_size, mean, std,
                                        args.onnx_provider, num_threads=nt)
            r["wall_clock_total_s"] = round(time.perf_counter() - t_start, 3)
            key = f"onnx_{nt_label}"
            all_results[key] = {"model_path": str(Path(args.onnx).resolve()), "result": r}
            print_result(f"ONNX ({args.onnx_provider}) [{nt_label}]", r)

        # ── Xác định danh sách file TFLite cần benchmark: ưu tiên --tflite-dir ──
        tflite_targets: list[Path] = []
        if args.tflite_dir:
            tflite_targets = sorted(Path(args.tflite_dir).glob("*.tflite"))
            if not tflite_targets:
                print(f"❌ Không tìm thấy file .tflite nào trong: {args.tflite_dir}")
                sys.exit(1)
        elif args.tflite:
            p = Path(args.tflite)
            if not p.exists():
                print(f"❌ Không tìm thấy file TFLite: {args.tflite}")
                sys.exit(1)
            tflite_targets = [p]

        if args.onnx and tflite_targets:
            print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s giữa ONNX và loạt TFLite để hệ thống ổn định lại...")
            time.sleep(PAUSE_BETWEEN_MODELS_S)

        # ── Benchmark từng file TFLite, có nghỉ giữa mỗi model ────────────
        for i, tflite_path in enumerate(tflite_targets):
            if i > 0:
                print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s trước khi benchmark model tiếp theo...")
                time.sleep(PAUSE_BETWEEN_MODELS_S)
            key = f"{tflite_path.stem}_{nt_label}"
            print(f"\n{'#'*70}\n# BENCHMARK TFLITE [{key}]: {tflite_path}\n{'#'*70}")
            t_start = time.perf_counter()
            r = benchmark_tflite_dataset(str(tflite_path), image_paths, labels, args.input_size, mean, std,
                                          num_threads=nt)
            r["wall_clock_total_s"] = round(time.perf_counter() - t_start, 3)
            all_results[key] = {"model_path": str(tflite_path.resolve()), "result": r}
            print_result(f"TFLite [{key}]", r)

        # ── Nghỉ giữa các cấu hình thread khác nhau (trừ lần cuối) ──────────
        if cfg_idx < len(thread_list) - 1:
            print(f"\n⏸  Nghỉ {PAUSE_BETWEEN_MODELS_S}s trước khi đổi sang cấu hình thread tiếp theo...")
            time.sleep(PAUSE_BETWEEN_MODELS_S)

    # ── So sánh nhanh: bảng tổng hợp toàn bộ (model x thread) ─────────
    print(f"\n{'='*70}\n  SO SÁNH TỔNG HỢP (tất cả model x tất cả cấu hình thread)\n{'='*70}")
    print(f"  {'Backend':<30} {'Latency(ms)':>14} {'FPS':>10} {'Acc(%)':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*10}")
    for name, entry in all_results.items():
        r = entry["result"]
        if "error" in r:
            print(f"  {name:<30} {'--- lỗi ---':>14}")
            continue
        print(f"  {name:<30} {r['latency_mean_ms']:>14} {r['throughput_fps']:>10} {r['accuracy']*100:>9.2f}")

    # ── Lưu báo cáo ──
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) / f"bench_dataset_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset_yaml": str(Path(args.dataset_yaml).resolve()),
            "num_classes": len(class_names),
            "num_images_total": len(image_paths),
            "input_size": args.input_size,
            "mean": mean, "std": std,
            "warmup_runs": WARMUP_RUNS,
            "pause_between_models_s": PAUSE_BETWEEN_MODELS_S,
            "num_threads_list": thread_list,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu báo cáo đầy đủ: {out_path}")


if __name__ == "__main__":
    main()