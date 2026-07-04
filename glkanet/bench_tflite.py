"""glkanet/bench_tflite.py — Đo FPS model .tflite trên Raspberry Pi.

Cài trên Pi (nhẹ hơn nhiều so với full tensorflow):
    pip install tflite-runtime --break-system-packages
    # nếu không có wheel cho Pi5/aarch64, fallback:
    pip install tensorflow --break-system-packages

Cách dùng:
    python bench_tflite.py --model weights/tflite/model_float32.tflite --input-size 224 --n 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback nếu không có tflite-runtime


def benchmark(model_path: str, input_size: int, n_runs: int, warmup: int, num_threads: int) -> None:
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_detail  = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    in_shape = input_detail["shape"]
    dtype    = input_detail["dtype"]

    print(f"[info] model:        {model_path}")
    print(f"[info] input shape:  {in_shape}, dtype={dtype}")
    print(f"[info] num_threads:  {num_threads}")

    if dtype == np.uint8 or dtype == np.int8:
        dummy = np.random.randint(0, 255, size=in_shape).astype(dtype)
    else:
        dummy = np.random.rand(*in_shape).astype(dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_detail["index"], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    fps = 1000.0 / times.mean()

    print(f"\n[result] n_runs={n_runs} warmup={warmup}")
    print(f"  mean:   {times.mean():.2f} ms")
    print(f"  std:    {times.std():.2f} ms")
    print(f"  p50:    {np.percentile(times, 50):.2f} ms")
    print(f"  p95:    {np.percentile(times, 95):.2f} ms")
    print(f"  min/max:{times.min():.2f} / {times.max():.2f} ms")
    print(f"  FPS:    {fps:.1f}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Benchmark FPS cho model .tflite trên Pi")
    ap.add_argument("--model",       required=True)
    ap.add_argument("--input-size",  type=int, default=224)
    ap.add_argument("--n",           type=int, default=200, help="Số lần đo")
    ap.add_argument("--warmup",      type=int, default=20)
    ap.add_argument("--num-threads", type=int, default=4, help="Số thread CPU (Pi5 có 4 core)")
    args = ap.parse_args()

    benchmark(
        model_path  = args.model,
        input_size  = args.input_size,
        n_runs      = args.n,
        warmup      = args.warmup,
        num_threads = args.num_threads,
    )


if __name__ == "__main__":
    _cli()
