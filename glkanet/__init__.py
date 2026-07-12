"""glkanet — GLKANet image classification library.

Quickstart:
    from glkanet import GLKA

    model = GLKA("simple_glka.yaml")
    model.train("configs/ccmt.yaml")
"""

# ── Main entry point (Giữ nguyên tên GLKA cho pipeline quản lý) ──
from glkanet.core    import GLKA

# ── Model & Utilities ──
from glkanet.builder import build_from_yaml, GLKANet
from glkanet.data    import get_data_loaders, DataConfig, ImageDataset
from glkanet.trainer import Trainer, evaluate, set_seed
from glkanet.exporter import export_all, load_checkpoint

# ── Blocks (Đưa toàn bộ 4 biến thể ra ngoài một cách an toàn bằng alias) ──
from glkanet.blocks  import (
    BLOCK_REGISTRY, 
    SEBlock, 
    Dual_Attention_Block,
    ShuffleGLKABlock,
    GLKA_CBAM as GLKA_CBAM_Block,
    GLKA_Shuffle as GLKA_Shuffle_Block,
    ECABlock,   
)

__version__ = "1.0.0"

__all__ = [
    # ── Main entry point ──
    "GLKA",
    # ── Model ──
    "build_from_yaml",
    "GLKANet",
    # ── Data ──
    "get_data_loaders",
    "DataConfig",
    "ImageDataset",
    # ── Training ──
    "Trainer",
    "evaluate",
    "set_seed",
    # ── Export ──
    "export_all",
    "load_checkpoint",
    # ── Blocks ──
    "BLOCK_REGISTRY",
    "SEBlock",
    "Dual_Attention_Block",
    "ShuffleGLKABlock",
    "GLKA_CBAM_Block",
    "GLKA_Shuffle_Block",
    "ECABlock",
]