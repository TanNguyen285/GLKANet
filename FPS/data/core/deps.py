"""
Auto-install các package cần thiết và expose các flag HAS_*.
Import file này TRƯỚC khi import PyQt5 để tránh DLL conflict trên Windows.
"""
import sys
import importlib
import subprocess

# ── auto-install helper ───────────────────────────────────────────────────────
def ensure(pkg: str, import_as: str = None):
    mod = import_as or pkg.split("[")[0].replace("-", "_")
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[setup] pip install {pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


# ── cài psutil trước mọi thứ ─────────────────────────────────────────────────
ensure("psutil")

# ── torch phải import TRƯỚC PyQt5 (Windows DLL conflict) ─────────────────────
HAS_TORCH        = False
HAS_THOP         = False
HAS_ONNX         = False
HAS_ONNXRT       = False
HAS_ONNX_OPCOUNTER = False

try:
    import torch                          # noqa: F401
    HAS_TORCH = True
    try:
        from thop import profile as thop_profile  # noqa: F401
        HAS_THOP = True
    except ImportError:
        pass
except ImportError:
    pass

try:
    import onnx                           # noqa: F401
    HAS_ONNX = True
except ImportError:
    pass

try:
    import onnxruntime                    # noqa: F401
    HAS_ONNXRT = True
except ImportError:
    pass

try:
    from onnx_opcounter import calculate_macs  # noqa: F401
    HAS_ONNX_OPCOUNTER = True
except ImportError:
    pass

# ── PyQt5 cài sau cùng ───────────────────────────────────────────────────────
ensure("PyQt5")

# ── auto-install các package tính năng nếu thiếu ─────────────────────────────
# Chạy sau PyQt5 để không block UI startup
if not HAS_THOP:
    ensure("thop")
if not HAS_ONNX:
    ensure("onnx")
if not HAS_ONNXRT:
    ensure("onnxruntime")
if not HAS_ONNX_OPCOUNTER:
    ensure("onnx-opcounter", import_as="onnx_opcounter")
    # Re-check sau khi cài
    try:
        from onnx_opcounter import calculate_macs  # noqa: F401
        HAS_ONNX_OPCOUNTER = True
    except ImportError:
        pass