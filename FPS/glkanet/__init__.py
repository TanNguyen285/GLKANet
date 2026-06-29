"""glkanet — GLKANet image classification library.

Quickstart:
    from glkanet import GLKA

    model = GLKA("simple_glka.yaml")
    model.train("configs/ccmt.yaml")
    model.val("configs/ccmt.yaml", split="test")
    model.export()

    model = GLKA.from_checkpoint("runs/exp1/weights/best_train.pt",
                                  "simple_glka.yaml")
    indices, names = model.predict(["img.jpg"])
"""

from glkanet.core    import GLKA
from glkanet.builder import build_from_yaml, GLKANet
from glkanet.data    import get_data_loaders, DataConfig, ImageDataset
from glkanet.trainer import Trainer, evaluate, set_seed
from glkanet.exporter import export_all, load_checkpoint
from glkanet.blocks  import BLOCK_REGISTRY, GLKA as GLKABlock, SEBlock, EfficientBlock

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
    "GLKABlock",
    "SEBlock",
    "EfficientBlock",
]
