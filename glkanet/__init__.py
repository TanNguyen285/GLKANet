from glkanet.core import GLKA

# Expose block classes để người dùng extend dễ
from glkanet.blocks import BLOCK_REGISTRY, GLKA as GLKABlock, SEBlock, EfficientBlock
from glkanet.builder import build_from_yaml, GLKANet

__version__ = "1.0.0"

__all__ = [
    # Main class
    "GLKA",
    # Builder
    "build_from_yaml",
    "GLKANet",
    # Blocks
    "BLOCK_REGISTRY",
    "GLKABlock",
    "SEBlock",
    "EfficientBlock",
]
