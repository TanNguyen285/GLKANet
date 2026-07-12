"""glkanet/blocks/__init__.py — Đăng ký các block kiến trúc mạng."""

from .conv_blocks import ConvBnRelu, conv_bn_relu
from .se_block import SEBlock
from .Attention_block import Dual_Attention_Block 
from .shuffle_block import ShuffleGLKABlock
from .eca_block import ECABlock

# Import khối đại diện không gian lớn gốc và các biến thể
from .glka_block_base import GLKA_CBAM
from .glka_shuffle import GLKA_Shuffle

# Registry map tên chuỗi từ YAML sang Class tương ứng
BLOCK_REGISTRY: dict[str, type] = {
    "ConvBnRelu": ConvBnRelu,
    "SEBlock": SEBlock,
    "Dual_Attention_Block": Dual_Attention_Block,
    "ShuffleGLKABlock": ShuffleGLKABlock,
    "GLKA_CBAM": GLKA_CBAM,
    "GLKA_Shuffle": GLKA_Shuffle,
    "ECABlock": ECABlock,
}

__all__ = [
    "ConvBnRelu",
    "conv_bn_relu",
    "SEBlock",
    "Dual_Attention_Block",
    "ShuffleGLKABlock",
    "GLKA_CBAM",
    "GLKA_Shuffle",
    "BLOCK_REGISTRY",
    "ECABlock",
]