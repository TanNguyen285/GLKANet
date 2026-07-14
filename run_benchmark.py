
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from layer_benchmark import (
    bench_e2e_torch,
    bench_per_layer_torch,
    bench_per_op_profiler,
    total_params,
    get_flops_per_layer,
    write_csv,
)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
ROOT = Path(r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_augument\weights")
TRAIN_FULL_PT  = ROOT / "best_train_full.pt"    # TorchScript chưa freeze hẳn -> có thể per-layer
DEPLOY_FULL_PT = ROOT / "best_deploy_full.pt"   # TorchScript freeze/optimize -> chỉ end-to-end
TRAIN_PT       = ROOT / "best_train.pt"         # checkpoint state_dict -> per-layer chắc ăn
CFG_YAML       = ROOT / "dualattention_glkaV1.yaml"
ONNX_PATH      = ROOT / "best_deploy.onnx"

IMG_SIZE   = 224
BATCH_SIZE = 1
N_ITERS    = 500
N_WARMUP   = 500
CPU_THREAD_CONFIGS = [4, 8]
RUN_GPU = torch.cuda.is_available()
OUT_DIR = Path("bench_results")

# Các model có sẵn (torchvision / timm) muốn so sánh cùng model custom của bạn.
# --- TẠM TẮT: giờ chỉ đo model GLKA của bạn, phần này để dành so sánh sau. ---
# Muốn bật lại: set RUN_MODEL_ZOO = True bên dưới, và điền lại dict này.
RUN_MODEL_ZOO = False

TORCHVISION_MODELS = {
    # "mobilenet_v3_small": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_small"])
    #     .mobilenet_v3_small(weights=None),
    # "mobilenet_v3_large": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_large"])
    #     .mobilenet_v3_large(weights=None),
    # "shufflenet_v2_x1_0": lambda: __import__("torchvision.models", fromlist=["shufflenet_v2_x1_0"])
    #     .shufflenet_v2_x1_0(weights=None),
    # "shufflenet_v2_x0_5": lambda: __import__("torchvision.models", fromlist=["shufflenet_v2_x0_5"])
    #     .shufflenet_v2_x0_5(weights=None),
    # "mobilenet_v4_conv_small": lambda: __import__("timm").create_model(
    #     "mobilenetv4_conv_small", pretrained=False, num_classes=22
    # ),
}


# ══════════════════════════════════════════════════════════════
# LOAD MODEL — 3 nguồn
# ══════════════════════════════════════════════════════════════
def load_ts(path: Path, device: str):
    """TorchScript (.pt). Nếu đã freeze/fuse thì per-layer sẽ tự bỏ qua
    (layer_benchmark.bench_per_layer_torch phát hiện không có leaf module)."""
    m = torch.jit.load(str(path), map_location=device)
    m.eval()
    return m


def load_raw_model(cfg_path: Path, weight_path: Path, device: str) -> nn.Module:
    """Model custom (GLKA) — load qua glkanet.exporter, KHÔNG qua trace/freeze
    nên cây module luôn còn nguyên -> per-layer luôn đo được."""
    from glkanet.exporter import load_checkpoint
    return load_checkpoint(weight_path, cfg_path, device=device).to(device).eval()


def load_zoo_model(name: str, device: str) -> nn.Module:
    """Model có sẵn (torchvision/timm) để so sánh — xem TORCHVISION_MODELS."""
    if name not in TORCHVISION_MODELS:
        raise KeyError(f"Model '{name}' chưa có trong TORCHVISION_MODELS. "
                        f"Các model hiện có: {list(TORCHVISION_MODELS.keys())}")
    model = TORCHVISION_MODELS[name]()
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════
# 1 "PHIÊN" BENCH ĐẦY ĐỦ CHO 1 MODEL (dùng chung cho custom lẫn model zoo)
# ══════════════════════════════════════════════════════════════
def run_full_bench_for_model(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    tag: str,               # tên hiển thị, vd "mobilenet_v3_small", "glka_train"
    device_tag: str,        # vd "cpu_4t", "cpu_8t", "gpu"
    do_per_layer: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    """Trả về (summary_row, per_layer_rows, per_op_rows).
    per_layer_rows chỉ có nếu model còn cây nn.Module (eager / TS chưa freeze).
    per_op_rows chỉ có nếu model là ScriptModule (đã freeze/fuse, vd deploy_full)."""
    e2e = bench_e2e_torch(model, x, device, N_ITERS, N_WARMUP)

    is_scripted = isinstance(model, torch.jit.ScriptModule)
    if is_scripted:
        # ScriptModule (đặc biệt sau freeze) không hỗ trợ hook -> fvcore không chạy được,
        # và freeze() fold param thành constant nên params luôn = 0 -> bỏ qua, khỏi spam warning.
        total_gflops = float("nan")
        n_params = float("nan")
        print(f"  [i] '{tag}' là ScriptModule (đã freeze/script) -> bỏ qua FLOPs/params/per-layer.")
    else:
        _, total_gflops = get_flops_per_layer(model, x)
        n_params = total_params(model)

    e2e.update({
        "backend": tag,
        "device": device_tag,
        "n_params": n_params,
        "gflops": total_gflops,
    })
    print(f"  {tag:<28} [{device_tag}]: mean={e2e['mean_ms']:.3f}ms std={e2e['std_ms']:.3f}ms "
          f"p95={e2e['p95_ms']:.3f}ms fps={e2e['fps']:.1f} "
          f"params={e2e['n_params']:,} gflops={e2e['gflops']:.3f}")

    layer_rows = []
    per_op_rows = []
    if is_scripted:
        # Không hook được theo tên module -> dùng torch.profiler để lấy breakdown theo op/kernel
        per_op_rows = bench_per_op_profiler(model, x, device, n_iters=100)
    elif do_per_layer:
        layer_rows = bench_per_layer_torch(model, x, device, n_iters=100, compute_flops=True)

    return e2e, layer_rows, per_op_rows


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    summary = []

    devices_to_run = []
    for n_threads in CPU_THREAD_CONFIGS:
        devices_to_run.append(("cpu", n_threads))
    if RUN_GPU:
        devices_to_run.append(("cuda", None))
    else:
        print("[!] Không có CUDA -> chỉ chạy CPU.")

    for device, n_threads in devices_to_run:
        if device == "cpu":
            torch.set_num_threads(n_threads)
            device_tag = f"cpu_{n_threads}t"
            print(f"\n{'='*70}\nCPU — {n_threads} threads\n{'='*70}")
        else:
            device_tag = "gpu"
            print(f"\n{'='*70}\nGPU — {torch.cuda.get_device_name(0)}\n{'='*70}")

        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)

        # ---- (a) Model custom của bạn: TorchScript deploy (freeze -> đo theo OP qua profiler) ----
        if DEPLOY_FULL_PT.exists():
            m = load_ts(DEPLOY_FULL_PT, device)
            s, _, per_op_rows = run_full_bench_for_model(m, x, device, "glka_deploy_full_ts", device_tag,
                                                           do_per_layer=False)
            summary.append(s)
            if per_op_rows:
                write_csv(per_op_rows, OUT_DIR / f"per_op_glka_deploy_full_{device_tag}.csv")

        # ---- (b) Model custom: TorchScript train_full ----
        # --- TẠM TẮT: giờ chỉ cần đo deploy_full. Bật lại bằng cách bỏ comment. ---
        # if TRAIN_FULL_PT.exists():
        #     m = load_ts(TRAIN_FULL_PT, device)
        #     s, layer_rows, per_op_rows = run_full_bench_for_model(m, x, device, "glka_train_full_ts", device_tag)
        #     summary.append(s)
        #     if layer_rows:
        #         write_csv(layer_rows, OUT_DIR / f"per_layer_glka_train_full_{device_tag}.csv")
        #     if per_op_rows:
        #         write_csv(per_op_rows, OUT_DIR / f"per_op_glka_train_full_{device_tag}.csv")

        # ---- (c) Model custom: PyTorch eager ----
        # --- TẠM TẮT: giờ chỉ cần đo deploy_full. Bật lại bằng cách bỏ comment. ---
        # if TRAIN_PT.exists() and CFG_YAML.exists():
        #     m = load_raw_model(CFG_YAML, TRAIN_PT, device)
        #     s, layer_rows, _ = run_full_bench_for_model(m, x, device, "glka_eager", device_tag)
        #     summary.append(s)
        #     write_csv(layer_rows, OUT_DIR / f"per_layer_glka_eager_{device_tag}.csv")

        # ---- (d) Các model có sẵn để so sánh: mobilenet_v3, v4, shufflenet ----
        # (đang tắt qua RUN_MODEL_ZOO=False -> giờ chỉ đo GLKA, để dành so sánh sau)
        if RUN_MODEL_ZOO:
            for zoo_name in TORCHVISION_MODELS:
                try:
                    m = load_zoo_model(zoo_name, device)
                except Exception as e:
                    print(f"  [!] Bỏ qua model zoo '{zoo_name}': {e}")
                    continue
                s, layer_rows, _ = run_full_bench_for_model(m, x, device, zoo_name, device_tag)
                summary.append(s)
                write_csv(layer_rows, OUT_DIR / f"per_layer_{zoo_name}_{device_tag}.csv")

        # ---- (e) ONNX (chỉ end-to-end, không per-layer/flops qua fvcore) ----
        if ONNX_PATH.exists():
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                if device == "cpu":
                    so.intra_op_num_threads = n_threads
                    so.inter_op_num_threads = 1
                provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
                sess = ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=[provider])
                input_name = sess.get_inputs()[0].name
                x_np = x.detach().cpu().numpy()

                from layer_benchmark import timed_loop, stats_from_times_ms

                def step():
                    sess.run(None, {input_name: x_np})

                times = timed_loop(step, N_ITERS, N_WARMUP)
                s = stats_from_times_ms(times)
                s.update({"backend": "glka_onnx", "device": device_tag, "n_params": float("nan"),
                           "gflops": float("nan")})
                summary.append(s)
                write_csv([s], OUT_DIR / f"onnx_{device_tag}.csv")
                print(f"  glka_onnx                    [{device_tag}]: mean={s['mean_ms']:.3f}ms "
                      f"fps={s['fps']:.1f}")
            except Exception as e:
                print(f"  [!] Bỏ qua ONNX: {e}")

    write_csv(summary, OUT_DIR / "summary_end_to_end.csv")

    print(f"\n{'='*70}\nTỔNG KẾT (mean_ms thấp hơn = nhanh hơn)\n{'='*70}")
    for r in sorted(summary, key=lambda r: r["mean_ms"]):
        print(f"  {r['backend']:<28} {r['device']:<10} mean={r['mean_ms']:>8.3f}ms  "
              f"std={r['std_ms']:>6.3f}  p95={r['p95_ms']:>8.3f}ms  fps={r['fps']:>7.1f}")


if __name__ == "__main__":
    main()