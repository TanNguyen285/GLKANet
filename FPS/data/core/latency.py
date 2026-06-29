"""
Benchmark latency / FPS.

Hai backend:
  - benchmark_torch()  → PyTorch model (CPU / CUDA / MPS)
  - benchmark_onnx()   → ONNX Runtime session

Defaults warmup/runs:
  - GPU (CUDA/MPS) : warmup=200,  runs=1000
  - CPU / edge     : warmup=50,   runs=200
  → User có thể override qua tham số warmup/runs từ UI
"""
from __future__ import annotations
import time
from typing import Any

from .deps import HAS_TORCH, HAS_ONNXRT

# ── Defaults theo device ───────────────────────────────────────────────────────

def _default_runs(device_str: str) -> tuple[int, int]:
    """Trả về (warmup, runs) hợp lý theo device."""
    if device_str == "cuda":
        return 200, 1000
    return 50, 200          # cpu / mps / edge (Pi 5)


# ── PyTorch ───────────────────────────────────────────────────────────────────

def benchmark_torch(
    model:      Any,
    dummy:      Any,
    device_str: str,
    warmup:     int | None = None,
    runs:       int | None = None,
) -> dict:
    """
    Đo latency PyTorch.
    warmup/runs: None → dùng default theo device.
    Trả về: mean/min/max/p50/p95 (ms) và throughput (FPS).
    """
    if not HAS_TORCH:
        return {"benchmark_error": "PyTorch chưa cài"}

    import torch
    device = torch.device(device_str)

    _warmup, _runs = _default_runs(device_str)
    warmup = warmup if warmup is not None else _warmup
    runs   = runs   if runs   is not None else _runs

    times: list[float] = []
    try:
        with torch.no_grad():
            # warmup
            for _ in range(warmup):
                model(dummy)
            if device_str == "cuda":
                torch.cuda.synchronize(device)

            # benchmark
            for _ in range(runs):
                if device_str == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                model(dummy)
                if device_str == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000)

        return _stats(times, runs, warmup)

    except Exception as ex:
        return {"benchmark_error": str(ex)}


def measure_gpu_memory(model: Any, dummy: Any, device_str: str) -> dict:
    """Đo peak GPU memory (chỉ CUDA)."""
    if not HAS_TORCH or device_str != "cuda":
        return {}
    import torch
    device = torch.device(device_str)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        return {"gpu_mem_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6}
    except Exception as ex:
        return {"gpu_mem_error": str(ex)}


# ── ONNX Runtime ─────────────────────────────────────────────────────────────

def benchmark_onnx(
    sess:         Any,
    dummy_inputs: dict,
    warmup:       int | None = None,
    runs:         int | None = None,
    device_str:   str        = "cpu",
) -> dict:
    """
    Đo latency ONNX Runtime.
    sess        : onnxruntime.InferenceSession đã khởi tạo.
    dummy_inputs: dict {input_name: np.ndarray}
    device_str  : dùng để chọn default warmup/runs phù hợp.
    """
    if not HAS_ONNXRT:
        return {"ort_error": "onnxruntime chưa cài"}

    _warmup, _runs = _default_runs(device_str)
    warmup = warmup if warmup is not None else _warmup
    runs   = runs   if runs   is not None else _runs

    times: list[float] = []
    try:
        for _ in range(warmup):
            sess.run(None, dummy_inputs)
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, dummy_inputs)
            times.append((time.perf_counter() - t0) * 1000)
        return _stats(times, runs, warmup)
    except Exception as ex:
        return {"ort_error": str(ex)}


# ── helper ────────────────────────────────────────────────────────────────────

def _stats(times: list[float], runs: int, warmup: int) -> dict:
    ts   = sorted(times)
    mean = sum(times) / len(times)
    return {
        "latency_mean_ms":  round(mean, 3),
        "latency_min_ms":   round(min(times), 3),
        "latency_max_ms":   round(max(times), 3),
        "latency_p50_ms":   round(ts[len(ts) // 2], 3),
        "latency_p95_ms":   round(ts[int(len(ts) * 0.95)], 3),
        "throughput_fps":   round(1000.0 / mean, 2),
        "benchmark_runs":   runs,
        "benchmark_warmup": warmup,
    }