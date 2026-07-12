"""
layer_benchmark.py
--------------------
FILE LÕI (model-agnostic) — không chứa bất kỳ đường dẫn/model cụ thể nào.
Chỉ nhận vào một nn.Module (bất kỳ: model của bạn, mobilenet_v3, mobilenet_v4,
shufflenet_v2, ...) + input tensor, rồi đo:

    - Thời gian end-to-end (mean/median/std/min/max/p95/p99/iqr/fps)
    - Thời gian TỪNG LỚP (per leaf-module), fps từng lớp, % đóng góp
    - FLOPs từng lớp + tổng model (fvcore, chuẩn ~ chuẩn paper)
    - Số Parameters từng lớp + tổng model

File khác (vd run_benchmark.py) sẽ import module này, tự lo việc LOAD model
cụ thể (torchscript / state_dict / torchvision zoo / onnx) rồi gọi các hàm
ở đây để đo. Nhờ vậy 1 file này dùng lại được cho MỌI kiến trúc.

Cài đặt cần có:
    pip install torch numpy fvcore --break-system-packages
    (fvcore optional nếu bạn không cần flops, script tự bỏ qua nếu thiếu)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
# 1. THỐNG KÊ THỜI GIAN
# ══════════════════════════════════════════════════════════════
def stats_from_times_ms(times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "iqr_ms": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else float("inf"),
    }


def timed_loop(fn: Callable, n_iters: int, n_warmup: int, sync_fn: Optional[Callable] = None) -> list[float]:
    """Đo n_iters lần bằng perf_counter, warm-up trước để tránh nhiễu do
    cudnn autotune / cache lần đầu / JIT chưa ổn định."""
    for _ in range(n_warmup):
        fn()
        if sync_fn:
            sync_fn()

    times = []
    for _ in range(n_iters):
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _sync_fn_for(device: str) -> Optional[Callable]:
    return (lambda: torch.cuda.synchronize()) if device == "cuda" else None


# ══════════════════════════════════════════════════════════════
# 2. FLOPS + PARAMS (model-agnostic, dùng fvcore)
# ══════════════════════════════════════════════════════════════
def get_flops_per_layer(model: nn.Module, input_tensor: torch.Tensor) -> tuple[dict, float]:
    """
    Trả về (dict {tên_leaf_module: flops}, total_gflops).
    Dùng fvcore.FlopCountAnalysis — đo theo MAC rồi nhân 2 ra FLOPs thật
    (quy ước chuẩn trong hầu hết paper/benchmark).
    Nếu model đã bị TorchScript freeze/fuse (không còn cây module) thì
    fvcore không chạy được -> trả về dict rỗng, total = nan.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[WARN] Chưa cài fvcore (`pip install fvcore --break-system-packages`) "
              "-> bỏ qua đo FLOPs.")
        return {}, float("nan")

    try:
        model.eval()
        analyzer = FlopCountAnalysis(model, input_tensor)
        analyzer.unsupported_ops_warnings(False)
        analyzer.uncalled_modules_warnings(False)
        by_module = analyzer.by_module()  # {name: MACs}, name="" là toàn model
        total_flops = by_module.get("", analyzer.total()) * 2
        per_layer = {k: v * 2 for k, v in by_module.items() if k != ""}
        return per_layer, total_flops / 1e9
    except Exception as e:
        print(f"[WARN] Không đo được FLOPs (model có thể đã freeze/fuse): {e}")
        return {}, float("nan")


def get_params_per_layer(model: nn.Module) -> dict:
    """Số param RIÊNG của từng leaf-module (recurse=False, không cộng dồn con)."""
    return {
        name: sum(p.numel() for p in module.parameters(recurse=False))
        for name, module in model.named_modules()
        if len(list(module.children())) == 0
    }


def total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ══════════════════════════════════════════════════════════════
# 3. BENCH END-TO-END (áp dụng cho mọi nn.Module / TorchScript module)
# ══════════════════════════════════════════════════════════════
def bench_e2e_torch(model, x: torch.Tensor, device: str, n_iters: int, n_warmup: int) -> dict:
    def step():
        with torch.no_grad():
            model(x)

    times = timed_loop(step, n_iters, n_warmup, sync_fn=_sync_fn_for(device))
    return stats_from_times_ms(times)


# ══════════════════════════════════════════════════════════════
# 4. BENCH PER-LAYER (thời gian + fps + flops + params từng lớp)
# ══════════════════════════════════════════════════════════════
def bench_per_layer_torch(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    n_iters: int = 100,
    compute_flops: bool = True,
) -> list[dict]:
    """
    Chỉ dùng được nếu model còn nguyên cây module (KHÔNG phải TorchScript đã
    freeze/fuse). Với model bất kỳ (custom, mobilenet_v3, shufflenet_v2, ...)
    đều dùng chung được hàm này miễn model.named_modules() còn leaf module.

    Cách đo: bắt input thật của từng leaf-module qua forward_hook 1 lần,
    sau đó chạy lại riêng lẻ module đó nhiều lần để đo thời gian "thuần"
    (không lẫn thời gian của submodule cha/con khác).
    """
    captured: dict[str, tuple] = {}
    order: list[str] = []

    def make_hook(name):
        def hook(module, inp, out):
            if name not in captured:
                captured[name] = tuple(i.detach().clone() if torch.is_tensor(i) else i for i in inp)
                order.append(name)
        return hook

    leaves = {n: m for n, m in model.named_modules() if len(list(m.children())) == 0 and n != ""}
    if not leaves:
        print("  [!] Model không có leaf module (đã freeze/fuse hoặc là scripted model) "
              "-> BỎ QUA per-layer.")
        return []

    handles = [m.register_forward_hook(make_hook(n)) for n, m in leaves.items()]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    params_dict = get_params_per_layer(model)
    flops_dict, _ = get_flops_per_layer(model, x) if compute_flops else ({}, float("nan"))

    sync = _sync_fn_for(device)
    rows = []
    for name in order:
        module = leaves[name]
        inputs = captured[name]

        def step(module=module, inputs=inputs):
            with torch.no_grad():
                module(*inputs)

        try:
            times = timed_loop(step, n_iters, n_warmup=max(10, n_iters // 10), sync_fn=sync)
            s = stats_from_times_ms(times)
        except Exception as e:
            s = {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan"), "std_ms": float("nan"),
                 "min_ms": float("nan"), "max_ms": float("nan"), "p95_ms": float("nan"),
                 "p99_ms": float("nan"), "iqr_ms": float("nan"), "fps": float("nan")}
            print(f"    [!] Bỏ qua '{name}': {e}")

        flops = flops_dict.get(name, float("nan"))
        rows.append({
            "layer": name,
            "type": getattr(module, "original_name", module.__class__.__name__),
            "n_params": params_dict.get(name, 0),
            "gflops": flops / 1e9 if flops == flops else float("nan"),  # NaN-safe check
            **s,
        })

    total_ms = sum(r["mean_ms"] for r in rows if r["mean_ms"] == r["mean_ms"])  # bỏ NaN
    for r in rows:
        r["pct_of_total"] = (r["mean_ms"] / total_ms * 100) if total_ms > 0 and r["mean_ms"] == r["mean_ms"] else float("nan")
    rows.sort(key=lambda r: (r["mean_ms"] if r["mean_ms"] == r["mean_ms"] else -1), reverse=True)
    return rows


# ══════════════════════════════════════════════════════════════
# 4b. BENCH THEO OP (dùng cho ScriptModule đã freeze/fuse — KHÔNG có
#     cây module nên không hook được, nhưng torch.profiler vẫn "nhìn"
#     xuống tận ATen op/kernel nên vẫn tách được từng phép toán).
# ══════════════════════════════════════════════════════════════
def bench_per_op_profiler(
    model,
    x: torch.Tensor,
    device: str,
    n_iters: int = 100,
    n_warmup: int = 20,
) -> list[dict]:
    """
    Dùng cho model KHÔNG còn cây nn.Module (ScriptModule đã freeze/fuse,
    ví dụ best_deploy_full.pt). torch.profiler hoạt động ở tầng op/kernel
    (conv2d, batch_norm, addmm, relu, ...) nên vẫn ra được breakdown thời
    gian dù không có tên layer kiểu 'backbone_layers.x...' nữa.

    Trả về list[dict] mỗi dòng là 1 loại op, đã cộng dồn qua n_iters lần
    chạy rồi chia trung bình -> cùng đơn vị ms như các hàm bench khác.
    """
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sync = _sync_fn_for(device)

    # warm-up trước, KHÔNG tính vào profiler
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
            if sync:
                sync()

    with profile(activities=activities, record_shapes=False) as prof:
        with torch.no_grad():
            for _ in range(n_iters):
                model(x)
                if sync:
                    sync()

    events = prof.key_averages()

    # Tên thuộc tính thời gian đổi khác nhau tùy phiên bản torch -> dò tự động,
    # ưu tiên GPU nếu chạy CUDA, không có thì fallback CPU.
    def pick_time_us(e):
        for attr in (
            ("self_cuda_time_total" if device == "cuda" else None),
            ("cuda_time_total" if device == "cuda" else None),
            "self_cpu_time_total",
            "cpu_time_total",
        ):
            if attr and hasattr(e, attr):
                val = getattr(e, attr)
                if val:  # bỏ qua 0/None, thử attr kế tiếp nếu op này không có giá trị ở đây
                    return val
        return 0.0

    total_time_us = sum(pick_time_us(e) for e in events if e.key not in ("", "ProfilerStep*")) or 1.0

    rows = []
    for e in events:
        if e.key in ("", "ProfilerStep*"):
            continue
        time_us_total_all_iters = pick_time_us(e)   # tổng thời gian (us) cộng dồn qua n_iters lần forward
        total_ms_all_iters = time_us_total_all_iters / 1000.0
        mean_ms_per_call = total_ms_all_iters / e.count if e.count else float("nan")
        mean_ms_per_forward = total_ms_all_iters / n_iters  # thời gian TB của op này trong 1 lần forward

        rows.append({
            "op": e.key,
            "calls_per_forward": e.count / n_iters,
            "mean_ms_per_forward": mean_ms_per_forward,   # <-- CỘT THỜI GIAN CHÍNH, xem cột này
            "mean_ms_per_call": mean_ms_per_call,
            "total_ms_all_iters": total_ms_all_iters,     # tổng cộng dồn qua toàn bộ n_iters lần chạy
            "pct_of_total": (time_us_total_all_iters / total_time_us) * 100.0,
        })

    rows.sort(key=lambda r: r["mean_ms_per_forward"], reverse=True)
    return rows



def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"  [!] Không có dữ liệu để ghi -> bỏ qua {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.resolve()}")