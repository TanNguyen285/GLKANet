"""glkanet/blocks/__init__.py — Đăng ký các block kiến trúc mạng."""

from .conv_blocks import ConvBnRelu, conv_bn_relu
from .se_block import SEBlock
from .efficient_block import EfficientBlock
from .shuffle_block import ShuffleGLKABlock

# Import khối đại diện không gian lớn gốc và các biến thể
from .glka_block_base import GLKA_CBAM
from .glka_shuffle import GLKA_Shuffle
from .glka_SExCA import GLKA_SExCA
from .glka_CA_SE import GLKA_CA_SE

# Registry map tên chuỗi từ YAML sang Class tương ứng
BLOCK_REGISTRY: dict[str, type] = {
    "ConvBnRelu": ConvBnRelu,
    "SEBlock": SEBlock,
    "EfficientBlock": EfficientBlock,
    "ShuffleGLKABlock": ShuffleGLKABlock,
    "GLKA_CBAM": GLKA_CBAM,
    "GLKA_Shuffle": GLKA_Shuffle,
    "GLKA_SExCA": GLKA_SExCA,
    "GLKA_CA_SE": GLKA_CA_SE,
}

__all__ = [
    "ConvBnRelu",
    "conv_bn_relu",
    "SEBlock",
    "EfficientBlock",
    "ShuffleGLKABlock",
    "GLKA_CBAM",
    "GLKA_Shuffle",
    "GLKA_SExCA",
    "GLKA_CA_SE",
    "BLOCK_REGISTRY",
]