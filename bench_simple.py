"""
Benchmark ONNX + TFLite đơn giản — chạy CLI, 1 process duy nhất (không multiprocessing/
spawn) để tránh xung đột DLL/overlay gây access violation từng gặp phải với
bản PyQt5 + subprocess isolation trước đó.

Đánh đổi: nếu bản thân onnxruntime/tflite crash native thật sự (model hỏng nặng),
cả script sẽ chết theo — nhưng với model tự train, kiểm soát được, rủi ro
này thấp hơn nhiều so với việc debug xung đột DLL không hồi kết.

Chạy ONNX:
    pip install onnx onnxruntime numpy psutil --break-system-packages
    python bench_simple.py path/to/model.onnx --providers CPU CUDA --runs 200 --warmup 50

Chạy TFLite (trong venv-tflite — cần tensorflow hoặc tflite-runtime):
    pip install tensorflow numpy psutil --break-system-packages
    python bench_simple.py path/to/model.tflite --providers CPU --runs 200 --warmup 50

    # Nếu chỉ cài tflite-runtime (nhẹ hơn, không cần full tensorflow):
    pip install tflite-runtime numpy psutil --break-system-packages

Có thể benchmark ONNX + TFLite trong CÙNG 1 lệnh để so sánh trực tiếp:
    python bench_simple.py --onnx path/to/model.onnx --tflite path/to/model.tflite --runs 200

Lưu ý về provider/delegate:
    - Với .onnx: --providers nhận CPU / CUDA / TensorRT (chạy qua onnxruntime).
    - Với .tflite: --providers chỉ có ý nghĩa CPU / GPU / NNAPI (chạy qua tf.lite delegate).
      Nếu máy không hỗ trợ GPU-delegate/NNAPI, script sẽ báo lỗi rõ ràng thay vì tự fallback âm thầm.
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

_ONNX_PROVIDER_MAP = {
    "CPU": "CPUExecutionProvider",
    "CUDA": "CUDAExecutionProvider",
    "TensorRT": "TensorrtExecutionProvider",
}
_ONNX_DEFAULT_WARMUP_RUNS = {"CPU": (100, 500), "CUDA": (200, 1000), "TensorRT": (200, 1000)}

# TFLite: không có khái niệm ExecutionProvider như ORT, dùng delegate.
# Map tên "provider" người dùng gõ -> loại delegate tương ứng, để giữ CLI đồng nhất với ONNX.
_TFLITE_DELEGATE_CHOICES = ["CPU", "GPU", "NNAPI"]
_TFLITE_DEFAULT_WARMUP_RUNS = {"CPU": (100, 500), "GPU": (200, 1000), "NNAPI": (200, 1000)}


# ── Thống kê: mean/std/CI95 đúng cách (t-distribution cho n nhỏ) ───────────

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


def _stats(times_ms: list[float]) -> dict:
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
        "n": n,
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
    }


# ── Memory sampler (peak RSS song song lúc infer) ───────────────────────────

class _MemSampler:
    def __init__(self, interval_s: float = 0.02):
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


# ── Phân tích model: params (đơn giản, không tính MACs để tránh phức tạp) ──

def analyze_onnx(path: str) -> dict:
    import onnx

    out: dict = {"file_size_mb": round(os.path.getsize(path) / 1e6, 3)}
    proto = onnx.load(path)
    out["opset"] = ", ".join(f"{o.domain or 'ai.onnx'}:{o.version}" for o in proto.opset_import)
    out["total_nodes"] = len(proto.graph.node)

    total_params = 0
    for init in proto.graph.initializer:
        numel = 1
        for d in init.dims:
            numel *= d
        total_params += numel
    out["total_params"] = total_params
    return out


def _get_tflite_interpreter_cls():
    """Ưu tiên tflite_runtime (nhẹ) nếu có, fallback sang tensorflow.lite (venv-tflite thường có sẵn)."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate, "tflite_runtime"
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter, getattr(tf.lite.experimental, "load_delegate", None), "tensorflow"


def analyze_tflite(path: str) -> dict:
    Interpreter, _, backend_name = _get_tflite_interpreter_cls()

    out: dict = {"file_size_mb": round(os.path.getsize(path) / 1e6, 3), "tflite_backend": backend_name}
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()

    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    out["total_nodes"] = len(interp._get_ops_details()) if hasattr(interp, "_get_ops_details") else None
    out["inputs"] = [
        {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
        for d in in_details
    ]
    out["outputs"] = [
        {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
        for d in out_details
    ]

    # Ước lượng total_params bằng cách cộng numel của mọi tensor có dữ liệu cố định (weight/bias).
    # TFLite không tách rõ "initializer" như ONNX, nên đây chỉ là ước lượng dựa trên tensor detail.
    total_params = 0
    try:
        for i in range(interp.get_tensor_details().__len__() if False else 0):
            pass  # placeholder giữ chỗ, xem đoạn dưới để lấy đúng API
    except Exception:
        pass
    try:
        for td in interp.get_tensor_details():
            # Tensor có "buffer index" != 0 thường là weight cố định (heuristic, không chính xác 100%)
            if td.get("index", -1) not in [d["index"] for d in in_details] and \
               td.get("index", -1) not in [d["index"] for d in out_details]:
                shape = td.get("shape")
                if shape is not None and len(shape) > 0:
                    numel = 1
                    for d in shape:
                        numel *= int(d)
                    total_params += numel
    except Exception:
        total_params = None
    out["total_params_estimate"] = total_params
    return out


# ── Benchmark ONNX (onnxruntime) ────────────────────────────────────────────

def benchmark_onnx(path: str, provider_name: str, warmup: int, runs: int) -> dict:
    import onnxruntime as ort

    requested = _ONNX_PROVIDER_MAP[provider_name]
    available = ort.get_available_providers()
    if requested not in available:
        return {"error": f"{requested} không có sẵn. Available: {available}"}

    try:
        sess = ort.InferenceSession(path, providers=[requested])
    except Exception as ex:
        return {"error": f"Không tạo được session: {ex}"}

    actual_provider = sess.get_providers()[0]
    fallback_reason = None
    if actual_provider != requested:
        fallback_reason = f"Yêu cầu {requested} nhưng ORT tự chọn {actual_provider}"

    dtype_map = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
    }
    dummy_inputs = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = dtype_map.get(inp.type, np.float32)
        dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)

    try:
        out0 = sess.run(None, dummy_inputs)[0]
        numeric_issue = None
        if not np.all(np.isfinite(out0)):
            numeric_issue = "Output chứa NaN/Inf"
    except Exception as ex:
        return {"error": f"Chạy inference thất bại: {ex}"}

    mem = _MemSampler()
    times = []
    for _ in range(warmup):
        sess.run(None, dummy_inputs)
    mem.start()
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, dummy_inputs)
        times.append((time.perf_counter() - t0) * 1000)
    mem_result = mem.stop()

    result = _stats(times)
    result["backend"] = "onnx"
    result["provider_requested"] = requested
    result["provider_used"] = actual_provider
    result["warmup"] = warmup
    result["runs"] = runs
    result.update(mem_result)
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    if numeric_issue:
        result["numeric_issue"] = numeric_issue
    return result


# ── Benchmark TFLite (tf.lite.Interpreter / tflite_runtime) ────────────────

def benchmark_tflite(path: str, delegate_name: str, warmup: int, runs: int) -> dict:
    Interpreter, load_delegate, backend_name = _get_tflite_interpreter_cls()

    delegates = []
    fallback_reason = None
    if delegate_name == "GPU":
        try:
            # Tên thư viện delegate khác nhau theo OS — chỉ hỗ trợ tốt trên Linux/Android.
            # Trên Windows thường KHÔNG có GPU delegate cho TFLite (khác hẳn ONNX+CUDA).
            gpu_lib = {"linux": "libtensorflowlite_gpu_delegate.so"}.get(sys.platform, None)
            if gpu_lib is None or load_delegate is None:
                return {"error": "GPU delegate cho TFLite không khả dụng trên hệ điều hành này "
                                  "(thường chỉ hỗ trợ Linux/Android). Dùng --providers CPU thay thế."}
            delegates = [load_delegate(gpu_lib)]
        except Exception as ex:
            return {"error": f"Không load được GPU delegate: {ex}. Dùng --providers CPU thay thế."}
    elif delegate_name == "NNAPI":
        return {"error": "NNAPI delegate chỉ chạy trên Android, không áp dụng được khi benchmark trên PC. "
                          "Dùng --providers CPU thay thế."}

    try:
        if delegates:
            interp = Interpreter(model_path=path, experimental_delegates=delegates)
        else:
            interp = Interpreter(model_path=path)
        interp.allocate_tensors()
    except Exception as ex:
        return {"error": f"Không tạo được interpreter: {ex}"}

    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    dummy_inputs = {}
    for d in in_details:
        shape = [s if s > 0 else 1 for s in d["shape"]]
        dtype = d["dtype"]
        if np.issubdtype(dtype, np.floating):
            data = np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            # int8/uint8 quantized model: random trong range hợp lệ của dtype, tránh overflow.
            info = np.iinfo(dtype)
            data = np.random.randint(info.min, info.max + 1, size=shape).astype(dtype)
        else:
            data = np.zeros(shape, dtype=dtype)
        dummy_inputs[d["index"]] = data

    def _run_once():
        for idx, data in dummy_inputs.items():
            interp.set_tensor(idx, data)
        interp.invoke()
        return [interp.get_tensor(d["index"]) for d in out_details]

    try:
        out0 = _run_once()[0]
        numeric_issue = None
        if np.issubdtype(out0.dtype, np.floating) and not np.all(np.isfinite(out0)):
            numeric_issue = "Output chứa NaN/Inf"
    except Exception as ex:
        return {"error": f"Chạy inference thất bại: {ex}"}

    mem = _MemSampler()
    times = []
    for _ in range(warmup):
        _run_once()
    mem.start()
    for _ in range(runs):
        t0 = time.perf_counter()
        _run_once()
        times.append((time.perf_counter() - t0) * 1000)
    mem_result = mem.stop()

    result = _stats(times)
    result["backend"] = "tflite"
    result["tflite_backend_lib"] = backend_name
    result["delegate_requested"] = delegate_name
    result["warmup"] = warmup
    result["runs"] = runs
    result.update(mem_result)
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    if numeric_issue:
        result["numeric_issue"] = numeric_issue
    return result


# ── Cảnh báo bất thường ──────────────────────────────────────────────────

def build_warnings(bench: dict) -> list[str]:
    warns = []
    if "error" in bench:
        return [f"❌ {bench['error']}"]
    if bench.get("fallback_reason"):
        warns.append(f"⚠️ {bench['fallback_reason']}")
    if bench.get("numeric_issue"):
        warns.append(f"❌ {bench['numeric_issue']}")
    if bench.get("cv_percent", 0) > 30:
        warns.append(f"⚠️ CV = {bench['cv_percent']}% quá cao — kết quả không ổn định, "
                      f"nên tăng warmup/runs hoặc đóng bớt ứng dụng nền")
    if bench.get("memory_available") is False:
        warns.append("ℹ️ Chưa cài psutil nên không đo được RAM")
    if not warns:
        warns.append("✅ Không phát hiện bất thường.")
    return warns


def _print_bench_result(label: str, bench: dict) -> None:
    print(f"\n=== {label} ===")
    if "error" in bench:
        print(f"  ❌ LỖI: {bench['error']}")
        return
    print(f"  Latency mean       : {bench['latency_mean_ms']} ms (std {bench['latency_std_ms']}, CV {bench['cv_percent']}%)")
    print(f"  Latency P50/P90/P99: {bench['latency_p50_ms']} / {bench['latency_p90_ms']} / {bench['latency_p99_ms']} ms")
    print(f"  Throughput         : {bench['throughput_fps']} FPS")
    if bench.get("memory_available"):
        print(f"  Peak RSS memory    : {bench['peak_rss_mb']} MB (delta {bench['delta_rss_mb']} MB)")
    for w in build_warnings(bench):
        print(f"  {w}")


def _run_for_model(model_path: str, kind: str, providers: list[str], warmup_arg, runs_arg) -> tuple[dict, dict]:
    """kind: 'onnx' hoặc 'tflite'. Trả về (analysis, results_by_provider)."""
    print(f"\n=== Phân tích model ({kind}): {model_path} ===")
    if kind == "onnx":
        analysis = analyze_onnx(model_path)
    else:
        analysis = analyze_tflite(model_path)
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    results = {}
    for provider_name in providers:
        default_table = _ONNX_DEFAULT_WARMUP_RUNS if kind == "onnx" else _TFLITE_DEFAULT_WARMUP_RUNS
        warmup, runs = default_table.get(provider_name, (50, 200))
        if warmup_arg is not None:
            warmup = warmup_arg
        if runs_arg is not None:
            runs = runs_arg

        print(f"\n>>> Benchmark {kind.upper()} | {provider_name} (warmup={warmup}, runs={runs})")
        if kind == "onnx":
            bench = benchmark_onnx(model_path, provider_name, warmup, runs)
        else:
            bench = benchmark_tflite(model_path, provider_name, warmup, runs)
        results[provider_name] = bench
        _print_bench_result(f"{kind.upper()} / {provider_name}", bench)

    return analysis, results


def main():
    ap = argparse.ArgumentParser(description="Benchmark ONNX và/hoặc TFLite, 1 process, không multiprocessing.")
    ap.add_argument("model_path", nargs="?", default=None,
                     help="Đường dẫn file .onnx hoặc .tflite (tự nhận diện qua đuôi file). "
                          "Bỏ qua nếu dùng --onnx/--tflite để so sánh cả 2.")
    ap.add_argument("--onnx", default=None, help="Đường dẫn file .onnx (dùng khi muốn so sánh ONNX vs TFLite cùng lúc)")
    ap.add_argument("--tflite", default=None, help="Đường dẫn file .tflite (dùng khi muốn so sánh ONNX vs TFLite cùng lúc)")
    ap.add_argument("--providers", nargs="+", default=None,
                     help="Danh sách provider/delegate. ONNX: CPU CUDA TensorRT. "
                          "TFLite: CPU GPU NNAPI. Mặc định: CPU cho cả hai.")
    ap.add_argument("--warmup", type=int, default=None, help="Số lần warmup (mặc định: tự động theo provider)")
    ap.add_argument("--runs", type=int, default=None, help="Số lần đo (mặc định: tự động theo provider)")
    ap.add_argument("--out", default="./bench_results", help="Thư mục lưu báo cáo JSON")
    args = ap.parse_args()

    targets = []  # list of (path, kind)
    if args.onnx:
        targets.append((args.onnx, "onnx"))
    if args.tflite:
        targets.append((args.tflite, "tflite"))
    if args.model_path:
        ext = Path(args.model_path).suffix.lower()
        if ext == ".onnx":
            targets.append((args.model_path, "onnx"))
        elif ext == ".tflite":
            targets.append((args.model_path, "tflite"))
        else:
            print(f"❌ Không nhận diện được loại model từ đuôi file: {ext} "
                  f"(chỉ hỗ trợ .onnx / .tflite)")
            sys.exit(1)

    if not targets:
        print("❌ Chưa truyền model nào. Dùng: python bench_simple.py model.onnx "
              "hoặc python bench_simple.py model.tflite "
              "hoặc python bench_simple.py --onnx a.onnx --tflite b.tflite")
        sys.exit(1)

    for path, _ in targets:
        if not Path(path).exists():
            print(f"❌ Không tìm thấy file: {path}")
            sys.exit(1)

    all_results = {}
    for path, kind in targets:
        default_providers = ["CPU"]
        providers = args.providers if args.providers else default_providers
        # Lọc bỏ những provider không hợp lệ cho từng loại model, tránh lỗi khó hiểu.
        valid_choices = list(_ONNX_PROVIDER_MAP.keys()) if kind == "onnx" else _TFLITE_DELEGATE_CHOICES
        providers_for_kind = [p for p in providers if p in valid_choices]
        if not providers_for_kind:
            print(f"⚠️ Không có provider hợp lệ cho {kind} trong {providers}, dùng CPU mặc định.")
            providers_for_kind = ["CPU"]

        analysis, results = _run_for_model(path, kind, providers_for_kind, args.warmup, args.runs)
        all_results[kind] = {"model_path": str(Path(path).resolve()), "analysis": analysis, "providers": results}

    # So sánh nhanh nếu có cả 2 backend
    if "onnx" in all_results and "tflite" in all_results:
        print("\n=== So sánh nhanh ONNX vs TFLite (provider đầu tiên của mỗi bên) ===")
        onnx_first = next(iter(all_results["onnx"]["providers"].values()), {})
        tflite_first = next(iter(all_results["tflite"]["providers"].values()), {})
        if "error" not in onnx_first and "error" not in tflite_first:
            print(f"  ONNX   latency mean : {onnx_first.get('latency_mean_ms')} ms | "
                  f"{onnx_first.get('throughput_fps')} FPS")
            print(f"  TFLite latency mean : {tflite_first.get('latency_mean_ms')} ms | "
                  f"{tflite_first.get('throughput_fps')} FPS")
        else:
            print("  ⚠️ Một trong hai bên lỗi, xem chi tiết log ở trên.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(Path(p).stem for p, _ in targets)
    out_path = Path(args.out) / f"{stem}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), **all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu báo cáo: {out_path}")


if __name__ == "__main__":
    main()