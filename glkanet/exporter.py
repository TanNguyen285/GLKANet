"""glkanet/exporter.py — Export 3 bản: train / deploy / onnx, kèm validate reparam
tự động + chạy test trên test set. TFLite (convert + eval accuracy ONNX/TFLite) được
đẩy hoàn toàn sang subprocess venv-tflite (environment.py) để tránh xung đột môi trường
torch <-> tensorflow/onnx2tf trong cùng 1 process.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)


class _Wrapper(nn.Module):
    """Chỉ lấy logits — ONNX không hỗ trợ tuple output tốt."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def _get_logits(out) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _validate_reparam(
    model_train: nn.Module,
    model_deploy: nn.Module,
    input_size:  int,
    atol:        float = 1e-3,
    rtol:        float = 1e-3,
    n_samples:   int   = 4,
    verbose:     bool  = True,
) -> bool:
    model_train.eval()
    model_deploy.eval()

    dummy = torch.randn(n_samples, 3, input_size, input_size)

    out_train  = _get_logits(model_train(dummy))
    out_deploy = _get_logits(model_deploy(dummy))

    max_abs_diff = (out_train - out_deploy).abs().max().item()
    ok = torch.allclose(out_train, out_deploy, atol=atol, rtol=rtol)

    if verbose:
        status = "OK" if ok else "MISMATCH"
        print(f"  [validate] train vs deploy logits: max_abs_diff={max_abs_diff:.6f} -> {status}")
        if not ok:
            print(f"  [validate] CẢNH BÁO: reparam/fold không tương đương! "
                  f"Kiểm tra lại switch_to_deploy() (BN fold, shuffle fold, reparam K x K).")

    return ok


@torch.no_grad()
def _validate_onnx(
    onnx_path:   Path,
    model_deploy: nn.Module,
    input_size:  int,
    atol:        float = 1e-3,
    rtol:        float = 1e-3,
    n_samples:   int   = 4,
    verbose:     bool  = True,
) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        if verbose:
            print("  [validate] onnxruntime chưa cài — bỏ qua bước validate ONNX "
                  "(pip install onnxruntime --break-system-packages để bật).")
        return True

    dummy = torch.randn(n_samples, 3, input_size, input_size)
    out_torch = _get_logits(model_deploy(dummy)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = sess.run(None, {"images": dummy.numpy()})[0]

    max_abs_diff = float(np.abs(out_torch - out_onnx).max())
    ok = np.allclose(out_torch, out_onnx, atol=atol, rtol=rtol)

    if verbose:
        status = "OK" if ok else "MISMATCH"
        print(f"  [validate] deploy vs onnx logits:  max_abs_diff={max_abs_diff:.6f} -> {status}")
        if not ok:
            print("  [validate] CẢNH BÁO: ONNX export lệch so với PyTorch deploy model!")

    return ok


# ═══════════════════════════════════════════════════════════════════════
#  Eval torch trên test set (giữ trong process chính vì cần torch + DataLoader)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_torch(model_deploy: nn.Module, test_loader: Iterable) -> tuple[float, float]:
    correct, total = 0, 0
    t0 = time.perf_counter()
    for batch in test_loader:
        images, labels = batch[0], batch[1]
        logits = _get_logits(model_deploy(images)).numpy()
        preds = logits.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.shape[0]
    elapsed = time.perf_counter() - t0
    return correct / total, elapsed / max(total, 1) * 1000  # (acc, ms/ảnh)


# ═══════════════════════════════════════════════════════════════════════
#  TFLite: convert + eval accuracy — chạy toàn bộ trong subprocess venv-tflite
# ═══════════════════════════════════════════════════════════════════════

def _run_tflite_pipeline(
    onnx_path:  Path,
    weights_dir: Path,
    input_size: int,
    test_dir:   str | None,
    class_names: list[str] | None,
    tflite_project_dir: str | Path,
    mode:       str = "all",
    n_calib:    int = 200,
    mean:       tuple[float, float, float] = DEFAULT_MEAN,
    std:        tuple[float, float, float] = DEFAULT_STD,
    verbose:    bool = True,
) -> tuple[Path | None, dict]:
    """Gọi environment.py (venv-tflite riêng) qua subprocess để:
      1) convert ONNX -> TFLite (fp32/fp16/int8), dùng đúng mean/std lúc train để calibrate
      2) eval accuracy ONNX + mọi .tflite trên test_dir thật, cùng mean/std đó
    Không raise nếu lỗi — chỉ log cảnh báo, để không làm gãy pipeline train/export
    ONNX vốn đã chạy xong ở bước trước.

    Returns:
        (tflite_dir hoặc None, backend_eval dict — rỗng nếu lỗi/không có test_dir)
    """
    tflite_dir = weights_dir / "tflite"
    tflite_dir.mkdir(parents=True, exist_ok=True)

    env_script = Path(tflite_project_dir) / "environment.py"
    if not env_script.exists():
        if verbose:
            print(f"  [tflite] Không tìm thấy {env_script} — bỏ qua bước TFLite.")
        return None, {}

    cmd = [
        sys.executable, str(env_script),
        "--onnx", str(onnx_path),
        "--out", str(tflite_dir),
        "--input-size", str(input_size),
        "--mode", mode,
        "--n-calib", str(n_calib),
        "--mean", ",".join(str(x) for x in mean),
        "--std", ",".join(str(x) for x in std),
    ]
    if test_dir:
        # Dùng luôn test_dir làm calib_dir vì đã có ảnh thật, khỏi cần tách riêng.
        cmd += ["--calib-dir", test_dir, "--test-dir", test_dir]
    if class_names:
        cmd += ["--class-names", ",".join(class_names)]

    if verbose:
        print(f"  [tflite] Gọi subprocess venv-tflite:\n           {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        if verbose:
            print("  [tflite] CẢNH BÁO: subprocess thất bại (bỏ qua, không làm gãy export).")
            if result.stdout:
                print(result.stdout[-3000:])
            if result.stderr:
                print(result.stderr[-3000:])
        return tflite_dir, {}

    eval_json = tflite_dir / "backend_eval_results.json"
    backend_eval = {}
    if eval_json.exists():
        try:
            backend_eval = json.loads(eval_json.read_text())
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [tflite] CẢNH BÁO: không đọc được {eval_json}: {e}")

    return tflite_dir, backend_eval


# ═══════════════════════════════════════════════════════════════════════
#  Export pipeline chính
# ═══════════════════════════════════════════════════════════════════════

def export_all(
    model:        nn.Module,
    save_dir:     Path,
    yaml_path:    str | Path,
    input_size:   int   = 224,
    opset:        int   = 18,
    verbose:      bool  = True,
    validate:     bool  = True,
    atol:         float = 1e-3,
    rtol:         float = 1e-3,
    # ── TFLite (subprocess venv-tflite) ─────────────────────
    export_tflite: bool = True,
    test_dir:      str | None = None,
    class_names:   list[str] | None = None,
    tflite_project_dir: str | Path = "TFlite",
    tflite_mode:   str = "all",
    n_calib:       int = 200,
    mean:          tuple[float, float, float] = DEFAULT_MEAN,
    std:           tuple[float, float, float] = DEFAULT_STD,
    # ── Test torch (process chính) ──────────────────────────
    test_loader:   Iterable | None = None,
) -> dict[str, Path | dict | bool | None]:
    """Xuất train / deploy / onnx, kèm validate, rồi (nếu export_tflite=True) đẩy
    ONNX sang subprocess venv-tflite để convert TFLite + eval accuracy ONNX/TFLite
    trên test_dir thật (dùng đúng mean/std normalize như lúc train). Nếu test_loader
    được truyền, tự chạy accuracy torch trong process chính và gộp chung vào
    test_results.

    Args:
        model:         GLKANet ở eval mode
        save_dir:      thư mục exp (weights/ sẽ tạo bên trong)
        yaml_path:     path file yaml kiến trúc — bắt buộc, tự copy vào weights/
        input_size:    chiều ảnh vuông
        opset:         ONNX opset (>= 18)
        verbose:       in log
        validate:      bật/tắt validate train-vs-deploy-vs-onnx
        atol/rtol:     ngưỡng sai số logits fp32
        export_tflite: bật/tắt convert + eval TFLite qua subprocess
        test_dir:      thư mục ảnh test THẬT dạng ImageFolder — bắt buộc nếu muốn
                       eval accuracy ONNX/TFLite
        class_names:   list tên class đúng thứ tự index model output
        tflite_project_dir: path tới thư mục chứa environment.py + onnx_to_tflite.py
        tflite_mode:   "fp32" | "fp16" | "int8" | "all"
        n_calib:       số ảnh dùng calibrate int8
        mean/std:      normalize PHẢI khớp transforms.Normalize() lúc train
                       (đọc từ train.yaml -> normalize.mean/std)
        test_loader:   DataLoader test THẬT — nếu truyền vào, tự chạy accuracy torch
                       trong process chính, gộp chung bảng kết quả với onnx/tflite

    Returns:
        {"train", "deploy", "onnx", "tflite_dir": Path|None, "yaml": Path,
         "validated": bool, "test_results": dict | None}
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml_path không tồn tại: {yaml_path}")

    weights_dir = Path(save_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # ── 1. Train weights (chưa reparam) ──────────────────────
    path_train = weights_dir / "best_train.pt"
    torch.save({"state_dict": model.state_dict(), "deployed": False}, path_train)
    if verbose:
        print(f"  [export] train   → {path_train.name}")

    # ── 2. Deploy weights (đã reparam) ───────────────────────
    model_deploy = copy.deepcopy(model).cpu()
    model_deploy.eval()
    model_deploy.switch_to_deploy()
    model_deploy.eval()

    path_deploy = weights_dir / "best_deploy.pt"
    torch.save({"state_dict": model_deploy.state_dict(), "deployed": True}, path_deploy)
    if verbose:
        print(f"  [export] deploy  → {path_deploy.name}")

    # ── 2b. Validate reparam ─────────────────────────────────
    all_valid = True
    if validate:
        model_cpu = copy.deepcopy(model).cpu()
        ok_reparam = _validate_reparam(
            model_cpu, model_deploy, input_size, atol=atol, rtol=rtol, verbose=verbose,
        )
        all_valid &= ok_reparam

    # ── 3. ONNX ──────────────────────────────────────────────
    path_onnx = weights_dir / "best_deploy.onnx"

    wrapper = _Wrapper(model_deploy).cpu()
    wrapper.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)

    dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        wrapper, dummy, str(path_onnx),
        opset_version=max(opset, 18),
        input_names=["images"], output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    if verbose:
        print(f"  [export] onnx    → {path_onnx.name}")

    try:
        import onnxsim
        import onnx as onnx_lib
        model_raw = onnx_lib.load(str(path_onnx))
        model_simplified, check_ok = onnxsim.simplify(model_raw)
        if check_ok:
            onnx_lib.save(model_simplified, str(path_onnx))
            if verbose:
                n_before = len(model_raw.graph.node)
                n_after  = len(model_simplified.graph.node)
                print(f"  [onnxsim] {n_before} nodes → {n_after} nodes "
                      f"(-{n_before - n_after}, {(1 - n_after/n_before)*100:.0f}%)")
        elif verbose:
            print("  [onnxsim] CẢNH BÁO: simplify check thất bại, giữ nguyên bản chưa simplify.")
    except ImportError:
        if verbose:
            print("  [onnxsim] chưa cài — bỏ qua (pip install onnxsim --break-system-packages).")

    if validate:
        ok_onnx = _validate_onnx(
            path_onnx, model_deploy, input_size, atol=atol, rtol=rtol, verbose=verbose,
        )
        all_valid &= ok_onnx

    # ── 4. TFLite (subprocess venv-tflite: convert + eval accuracy) ──
    tflite_dir = None
    backend_eval: dict = {}
    if export_tflite:
        tflite_dir, backend_eval = _run_tflite_pipeline(
            onnx_path=path_onnx,
            weights_dir=weights_dir,
            input_size=input_size,
            test_dir=test_dir,
            class_names=class_names,
            tflite_project_dir=tflite_project_dir,
            mode=tflite_mode,
            n_calib=n_calib,
            mean=mean,
            std=std,
            verbose=verbose,
        )

    # ── 5. Copy yaml kiến trúc ────────────────────────────────
    path_yaml = weights_dir / yaml_path.name
    shutil.copy2(yaml_path, path_yaml)
    if verbose:
        print(f"  [export] yaml    → {path_yaml.name}")

    if verbose:
        _print_sizes(path_train, path_deploy, path_onnx, path_yaml)
        if validate:
            print(f"  [export] validate tổng: {'OK' if all_valid else 'CÓ MISMATCH — xem log ở trên'}")

    # ── 6. Test torch (process chính) + gộp kết quả onnx/tflite từ subprocess ──
    test_results = None
    if test_loader is not None or backend_eval:
        test_results = {}
        if test_loader is not None:
            model_deploy.eval()
            acc_torch, ms_torch = _eval_torch(model_deploy, test_loader)
            test_results["torch"] = {"acc": acc_torch, "ms_per_img": ms_torch}
        test_results.update(backend_eval)

        if verbose:
            print("\n  [test] ═══ Kết quả trên test set (torch + onnx/tflite qua subprocess) ═══")
            print(f"  {'Backend':<25} {'Accuracy':>10} {'ms/ảnh':>10}")
            print(f"  {'-'*25} {'-'*10} {'-'*10}")
            for name, r in test_results.items():
                if isinstance(r, dict) and "acc" in r:
                    print(f"  {name:<25} {r['acc']*100:>9.2f}% {r['ms_per_img']:>9.2f}")
                else:
                    print(f"  {name:<25} {'--- lỗi/không có':>21}")

    return {
        "train":        path_train,
        "deploy":       path_deploy,
        "onnx":         path_onnx,
        "tflite_dir":   tflite_dir,
        "yaml":         path_yaml,
        "validated":    all_valid,
        "test_results": test_results,
    }


def _print_sizes(*paths: Path | None) -> None:
    print("\n  [export] File sizes:")
    for p in paths:
        if p is not None and p.exists():
            print(f"           {p.name:<25} {p.stat().st_size / 1024 / 1024:.2f} MB")


def load_checkpoint(
    pt_path:   str | Path,
    yaml_path: str | Path,
    device:    str = "cpu",
):
    try:
        from glkanet.builder import build_from_yaml
    except ImportError:
        from builder import build_from_yaml

    ckpt  = torch.load(pt_path, map_location=device, weights_only=True)
    model = build_from_yaml(yaml_path)
    if ckpt.get("deployed", False):
        model.switch_to_deploy()

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device)