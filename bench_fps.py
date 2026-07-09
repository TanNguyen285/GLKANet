from __future__ import annotations
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.benchmark as torch_benchmark
from torchvision import models as tvm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from glkanet import GLKA

NUM_CLASSES = 22
IMG_SIZE    = 224

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"  # đổi thành "cpu" nếu muốn ép benchmark CPU (vd để so với Pi)
NUM_THREADS  = 4         # chỉ có tác dụng khi DEVICE="cpu" — số threads cố định để tái lập kết quả
MIN_RUN_TIME = 2.0       # giây, càng lớn càng ổn định nhưng càng lâu
N_WARMUP     = 10        # số lần forward warmup trước khi đo (ngoài warmup nội bộ của blocked_autorange)

RUNS_DIR   = Path("runs_compare")
TRAIN_CSV  = RUNS_DIR / "results_train.csv"   # do train_compare.py sinh ra (acc/f1/params)

GLKANET_ARCH_YAML  = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\glkanet\configs\Hybird.yaml"
GLKANET_CHECKPOINT = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\hybird\weights\best_deploy.pt"


def build_shufflenet_v2(num_classes):
    m = tvm.shufflenet_v2_x0_5(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_mobilenet_v3_small(num_classes):
    m = tvm.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m

def _build_timm(name, num_classes):
    return timm.create_model(name, pretrained=False, num_classes=num_classes)

MODEL_REGISTRY = {
    "ShuffleNetV2_x0.5":        build_shufflenet_v2,
    "MobileNetV3_small":        build_mobilenet_v3_small,
}
if HAS_TIMM:
    MODEL_REGISTRY.update({
        "MobileNetV4_conv_s050": lambda nc: _build_timm("mobilenetv4_conv_small_050", nc),
        "RepViT_m0.9":           lambda nc: _build_timm("repvit_m0_9", nc),
        "FastViT_t8":            lambda nc: _build_timm("fastvit_t8", nc),
        "EdgeNeXt_xx_small":     lambda nc: _build_timm("edgenext_xx_small", nc),
        "StarNet_s1":            lambda nc: _build_timm("starnet_s1", nc),
    })


class GLKANetLogitsOnly(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def load_glkanet_deployed(checkpoint_path, arch_yaml, device):
    wrapper = GLKA(arch_yaml)
    wrapper.load(checkpoint_path)
    model = GLKANetLogitsOnly(wrapper._model).to(device).eval()
    return model


def load_trained_checkpoint(name, builder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = builder(num_classes)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK — batch=1, num_threads cố định, warmup thủ công + blocked_autorange
# ══════════════════════════════════════════════════════════════════════
def bench_latency(model, device, img_size=IMG_SIZE, min_run_time=MIN_RUN_TIME, n_warmup=N_WARMUP):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(x)

        timer = torch_benchmark.Timer(
            stmt="model(x)", globals={"model": model, "x": x},
            num_threads=torch.get_num_threads(), label="latency",
        )
        measurement = timer.blocked_autorange(min_run_time=min_run_time)

    median_ms = measurement.median * 1000.0
    mean_ms   = measurement.mean * 1000.0
    iqr_ms    = measurement.iqr * 1000.0
    fps = 1000.0 / median_ms
    return median_ms, mean_ms, iqr_ms, fps


def main():
    torch.set_num_threads(NUM_THREADS)
    device = torch.device(DEVICE)
    print(f"[info] Device: {device}  |  num_threads: {torch.get_num_threads()}")

    # merge với kết quả train (params/acc/f1) nếu có
    train_rows = {}
    if TRAIN_CSV.exists():
        with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                train_rows[r["model"]] = r
    else:
        print(f"[warn] Không thấy {TRAIN_CSV} — chỉ đo FPS, không gộp acc/params.")

    results = []

    # ── GLKANet ──
    print(f"\n{'='*100}\n[FPS] GLKANet\n{'='*100}")
    try:
        glka_model = load_glkanet_deployed(GLKANET_CHECKPOINT, GLKANET_ARCH_YAML, device)
        n_params = sum(p.numel() for p in glka_model.parameters())
        median_ms, mean_ms, iqr_ms, fps = bench_latency(glka_model, device)
        row = _make_row("GLKANet", n_params, median_ms, mean_ms, iqr_ms, fps, train_rows)
        results.append(row)
        _print_row(row)
    except Exception as ex:
        print(f"[warn] Bench GLKANet lỗi: {ex}")

    # ── Các model train từ checkpoint ──
    for name, builder in MODEL_REGISTRY.items():
        ckpt_path = RUNS_DIR / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[warn] Không thấy checkpoint {ckpt_path} — bỏ qua {name} (chạy train_compare.py trước).")
            continue

        print(f"\n{'='*100}\n[FPS] {name}\n{'='*100}")
        try:
            model = load_trained_checkpoint(name, builder, ckpt_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            median_ms, mean_ms, iqr_ms, fps = bench_latency(model, device)
            row = _make_row(name, n_params, median_ms, mean_ms, iqr_ms, fps, train_rows)
            results.append(row)
            _print_row(row)
        except Exception as ex:
            print(f"[warn] Bench '{name}' lỗi: {ex}")

    # ── In bảng tổng hợp + lưu ──
    print(f"\n{'='*100}\nBẢNG TỔNG HỢP FPS (num_threads={NUM_THREADS}, device={DEVICE})\n{'='*100}")
    header = (f"{'Model':<24}{'Params(M)':>11}{'TestAcc':>10}{'MacroF1':>10}"
              f"{'Median(ms)':>12}{'Mean(ms)':>11}{'IQR(ms)':>10}{'FPS':>9}{'FPS/M':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<24}{r['params_M']:>11}{r['test_acc']:>10}{r['macro_f1']:>10}"
              f"{r['latency_median_ms']:>12}{r['latency_mean_ms']:>11}{r['latency_iqr_ms']:>10}"
              f"{r['fps']:>9}{r['fps_per_M_params']:>9}")

    csv_path = RUNS_DIR / "results_fps.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    json_path = RUNS_DIR / "results_fps.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[info] Đã lưu: {csv_path}  và  {json_path}")


def _make_row(name, n_params, median_ms, mean_ms, iqr_ms, fps, train_rows):
    tr = train_rows.get(name, {})
    params_m = round(n_params / 1e6, 4)
    return {
        "model": name,
        "params_M": params_m,
        "test_acc": tr.get("test_acc", ""),
        "macro_f1": tr.get("macro_f1", ""),
        "latency_median_ms": round(median_ms, 4),
        "latency_mean_ms": round(mean_ms, 4),
        "latency_iqr_ms": round(iqr_ms, 4),
        "fps": round(fps, 2),
        "fps_per_M_params": round(fps / max(1e-9, params_m), 2),
    }


def _print_row(row):
    print(f"[{row['model']}] params={row['params_M']}M  median={row['latency_median_ms']}ms  "
          f"mean={row['latency_mean_ms']}ms  iqr={row['latency_iqr_ms']}ms  fps={row['fps']}")


if __name__ == "__main__":
    main()
