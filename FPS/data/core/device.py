"""
Detect toàn bộ thông tin thiết bị:
CPU, RAM, GPU (CUDA/MPS), board (Jetson / Raspberry Pi / Coral).
"""
import os
import platform
import subprocess

import psutil

from .deps import HAS_TORCH, HAS_ONNXRT


def detect_all() -> dict:
    """Trả về dict chứa mọi thông tin phần cứng + phần mềm."""
    info: dict = {}

    # ── System ───────────────────────────────────────────────────────────────
    info["os"]       = f"{platform.system()} {platform.release()}"
    info["arch"]     = platform.machine()
    info["hostname"] = platform.node()
    info["python"]   = platform.python_version()

    # ── CPU ──────────────────────────────────────────────────────────────────
    info["cpu_name"]           = _cpu_name()
    info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or "?"
    info["cpu_cores_logical"]  = psutil.cpu_count(logical=True)  or "?"
    info["cpu_freq"]           = _cpu_freq()

    # ── RAM ──────────────────────────────────────────────────────────────────
    mem = psutil.virtual_memory()
    info["ram_total"] = f"{mem.total / 1e9:.1f} GB"
    info["ram_avail"] = f"{mem.available / 1e9:.1f} GB"

    # ── Board (Jetson / RPi / Coral) ─────────────────────────────────────────
    info["board"] = _detect_board()

    # ── PyTorch / CUDA / MPS ─────────────────────────────────────────────────
    if HAS_TORCH:
        import torch
        info["torch_version"]  = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"]  = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append({
                    "name": p.name,
                    "vram": f"{p.total_memory / 1e9:.1f} GB",
                    "sm":   f"SM {p.major}.{p.minor}",
                    "mp":   p.multi_processor_count,
                })
        cudnn = getattr(torch.backends, "cudnn", None)
        info["cudnn_version"] = getattr(cudnn, "version", lambda: None)() if cudnn else None
    else:
        info["torch_version"] = None

    # ── ONNX Runtime ─────────────────────────────────────────────────────────
    if HAS_ONNXRT:
        import onnxruntime as ort
        info["onnxrt_version"]   = ort.__version__
        info["onnxrt_providers"] = ort.get_available_providers()
    else:
        info["onnxrt_version"] = None

    return info


def best_torch_device() -> str:
    """Trả về device string tốt nhất hiện có: 'cuda' | 'mps' | 'cpu'."""
    if not HAS_TORCH:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_torch_devices() -> list[dict]:
    """
    Trả về list các device UI có thể hiển thị dưới dạng radio button.

    Mỗi item: {"label": str, "value": str, "is_default": bool}

    Ví dụ:
      [
        {"label": "CPU  (Intel Core i7-13700H)",  "value": "cpu",  "is_default": False},
        {"label": "CUDA (NVIDIA RTX 4050 6GB)",   "value": "cuda", "is_default": True},
      ]
    """
    devices: list[dict] = []
    default = best_torch_device()

    # CPU — luôn có
    devices.append({
        "label":      f"CPU  ({_cpu_name()})",
        "value":      "cpu",
        "is_default": default == "cpu",
    })

    if HAS_TORCH:
        import torch

        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                vram_gb = p.total_memory / 1e9
                val = "cuda" if i == 0 else f"cuda:{i}"
                devices.append({
                    "label":      f"CUDA ({p.name}  {vram_gb:.0f} GB)",
                    "value":      val,
                    "is_default": default == "cuda" and i == 0,
                })

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append({
                "label":      "MPS  (Apple Silicon)",
                "value":      "mps",
                "is_default": default == "mps",
            })

    return devices


# ── helpers ───────────────────────────────────────────────────────────────────

def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            return (
                subprocess.check_output("wmic cpu get name", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
        except Exception:
            return platform.processor()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown"


def _cpu_freq() -> str:
    try:
        freq = psutil.cpu_freq()
        if freq:
            return f"{freq.current:.0f} MHz (max {freq.max:.0f} MHz)"
    except Exception:
        pass
    return "N/A"


def _detect_board() -> str | None:
    """Nhận diện Raspberry Pi, Jetson, Coral Dev Board."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path) as f:
                return f.read().strip().rstrip("\x00")
        except Exception:
            pass
    if os.path.exists("/etc/nv_tegra_release"):
        try:
            with open("/etc/nv_tegra_release") as f:
                line = f.readline().strip()
            return f"NVIDIA Jetson ({line})"
        except Exception:
            return "NVIDIA Jetson"
    if os.path.exists("/sys/bus/platform/devices/soc@0"):
        return "Google Coral Dev Board (maybe)"
    return None