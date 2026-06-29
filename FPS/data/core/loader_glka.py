"""
Loader riêng cho GLKANet checkpoint.

Format checkpoint (từ exporter.py):
  {"state_dict": OrderedDict, "deployed": bool}

Flow:
  1. Load .pt → đọc state_dict + deployed flag
  2. Tìm .yaml cùng thư mục (hoặc parent) với .pt
  3. build_from_yaml() → model skeleton
  4. load_state_dict() → khớp key/shape
  5. switch_to_deploy() nếu deployed=True
  6. Benchmark latency + FLOPs
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from .deps       import HAS_TORCH, HAS_THOP
from .model_size import count_params_from_state_dict, compute_flops, infer_input_size
from .latency    import benchmark_torch, measure_gpu_memory


# ── yaml finder ───────────────────────────────────────────────────────────────

def _find_yaml(pt_path: str) -> Path | None:
    """
    Tìm .yaml gần nhất theo thứ tự ưu tiên:
      1. Cùng thư mục với .pt (weights/)
      2. Thư mục cha (exp/)
      3. Thư mục cha của cha (run root)
    Nếu có nhiều .yaml → ưu tiên tên khớp với stem của .pt, rồi mới lấy cái đầu tiên.
    """
    pt = Path(pt_path).resolve()
    stem = pt.stem  # e.g. "best_deploy" hoặc "best_train"

    search_dirs = [
        pt.parent,                        # weights/
        pt.parent.parent,                 # exp1/
        pt.parent.parent.parent,          # runs/
        pt.parent.parent.parent.parent,   # Simple_GLKA/  ← yaml ở đây
    ]
    candidates: list[Path] = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(d.glob("*.yaml"))
            candidates.extend(d.glob("*.yml"))

    if not candidates:
        return None

    # ưu tiên yaml tên khớp stem
    for c in candidates:
        if c.stem == stem:
            return c
    return candidates[0]


# ── public loader ─────────────────────────────────────────────────────────────

def load_glka(
    pt_path:    str,
    device_str: str        = "cpu",
    yaml_path:  str | None = None,
    warmup:     int | None = None,
    runs:       int | None = None,
) -> dict:
    """
    Load GLKANet checkpoint + benchmark.

    Parameters
    ----------
    pt_path    : đường dẫn .pt / .pth
    device_str : 'cpu' | 'cuda' | 'mps'
    yaml_path  : nếu None → tự tìm .yaml cùng thư mục
    warmup     : None → dùng default theo device
    runs       : None → dùng default theo device

    Raises
    ------
    FileNotFoundError : không tìm được .yaml
    KeyError          : checkpoint không có key 'state_dict'
    RuntimeError      : load_state_dict thất bại (key/shape mismatch)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch chưa cài. pip install torch")

    import torch

    result: dict = {
        "format":       "GLKANet",
        "file_size_mb": os.path.getsize(pt_path) / 1e6,
    }

    # ── Bước 1: Load checkpoint ───────────────────────────────────────────
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as ex:
        raise RuntimeError(f"[load_glka] Không đọc được file .pt\n  → {ex}") from ex

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise KeyError(
            f"[load_glka] Checkpoint không có key 'state_dict'.\n"
            f"  Keys tìm thấy: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    sd       = ckpt["state_dict"]
    deployed = bool(ckpt.get("deployed", False))
    result["ck_type"]  = "glka_deploy" if deployed else "glka_train"
    result["deployed"] = deployed
    result.update(count_params_from_state_dict(sd))

    # ── Bước 2: Tìm yaml ─────────────────────────────────────────────────
    if yaml_path is None:
        found = _find_yaml(pt_path)
        if found is None:
            raise FileNotFoundError(
                f"[load_glka] Không tìm được .yaml gần '{pt_path}'.\n"
                f"  Đặt .yaml cùng thư mục với .pt hoặc thư mục cha."
            )
        yaml_path = str(found)

    result["yaml_file"] = os.path.basename(yaml_path)

    # ── Bước 3: Build model từ yaml ──────────────────────────────────────
    try:
        # Thêm project root vào sys.path để tìm glkanet/
        # Project root = thư mục chứa yaml (thường là Simple_GLKA/)
        import sys
        _yaml_dir = str(Path(yaml_path).resolve().parent)
        if _yaml_dir not in sys.path:
            sys.path.insert(0, _yaml_dir)

        try:
            from glkanet.builder import build_from_yaml
        except ImportError:
            from builder import build_from_yaml

        model = build_from_yaml(yaml_path)
    except ImportError as ex:
        raise RuntimeError(
            f"[load_glka] Không import được glkanet.builder.\n"
            f"  Đảm bảo glkanet đã cài (pip install -e .) hoặc nằm trong PYTHONPATH.\n"
            f"  → {ex}"
        ) from ex
    except Exception as ex:
        raise RuntimeError(
            f"[load_glka] build_from_yaml thất bại với '{os.path.basename(yaml_path)}'.\n"
            f"  → {ex}"
        ) from ex

    # ── Bước 4: switch_to_deploy TRƯỚC nếu checkpoint là deploy ─────────
    # deployed=True → branches đã bị fold thành reparam_conv trong sd
    # → phải switch model skeleton trước để key khớp rồi mới load
    if deployed:
        try:
            model.switch_to_deploy()
            result["reparam_applied"] = True
        except Exception as ex:
            raise RuntimeError(
                f"[load_glka] switch_to_deploy() thất bại.\n  → {ex}"
            ) from ex

    # ── Bước 5: Load state_dict ──────────────────────────────────────────
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except Exception as ex:
        raise RuntimeError(
            f"[load_glka] load_state_dict thất bại.\n  → {ex}"
        ) from ex

    # Phân loại mismatch
    model_sd       = model.state_dict()
    shape_mismatch = [
        k for k in (set(sd) & set(model_sd))
        if sd[k].shape != model_sd[k].shape
    ]

    if shape_mismatch:
        raise RuntimeError(
            f"[load_glka] Shape mismatch ở {len(shape_mismatch)} key(s).\n"
            f"  Ví dụ: {shape_mismatch[:3]}\n"
            f"  → .yaml không khớp với checkpoint này."
        )
    if len(missing) > len(sd) * 0.3:
        raise RuntimeError(
            f"[load_glka] Quá nhiều missing keys ({len(missing)}/{len(sd)}).\n"
            f"  Ví dụ: {list(missing[:3])}\n"
            f"  → .yaml có thể sai hoặc checkpoint bị hỏng."
        )
    if missing or unexpected:
        result["load_warning"] = (
            f"{len(missing)} missing / {len(unexpected)} unexpected keys "
            f"(thường do BN buffer — có thể bỏ qua)"
        )
    result["missing_keys_count"]  = len(missing)
    result["missing_keys_sample"] = list(missing[:5])

    # ── Bước 6: Forward pass + benchmark ─────────────────────────────────
    device = torch.device(device_str)
    try:
        model = model.float().eval().to(device)
        in_size = infer_input_size(sd)

        # Đọc input_size từ yaml nếu có
        try:
            import yaml as _yaml
            with open(yaml_path) as f:
                cfg = _yaml.safe_load(f)
            in_size = int(cfg.get("input_size", in_size))
        except Exception:
            pass

        dummy = torch.zeros(1, 3, in_size, in_size, device=device)
        result["input_size"] = in_size

        result.update(compute_flops(model, dummy))
        result.update(benchmark_torch(model, dummy, device_str, warmup, runs))
        result.update(measure_gpu_memory(model, dummy, device_str))

    except Exception as ex:
        result["profile_error"] = str(ex)

    return result


# ── detect helper (dùng trong analyzer.py) ───────────────────────────────────

def is_glka_checkpoint(pt_path: str) -> bool:
    """
    Kiểm tra nhanh xem checkpoint có phải format GLKANet không.
    Điều kiện: dict với key 'state_dict' và 'deployed'.
    Không load toàn bộ tensor — chỉ đọc metadata.
    """
    if not HAS_TORCH:
        return False
    import torch
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        return (
            isinstance(ckpt, dict)
            and "state_dict" in ckpt
            and "deployed" in ckpt
        )
    except Exception:
        return False