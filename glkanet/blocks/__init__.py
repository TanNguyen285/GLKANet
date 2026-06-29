"""Block registry — mọi block đều đăng ký ở đây để builder.py import."""

from .conv_blocks     import ConvBnRelu, conv_bn_relu
from .se_block        import SEBlock
from .glka_block      import GLKA
from .efficient_block import EfficientBlock

# -----------------------------------------------------------------------
# Registry: map tên string (từ yaml) → class
# Thêm block mới: chỉ cần import class và đăng ký tên vào đây
# -----------------------------------------------------------------------
BLOCK_REGISTRY: dict = {
    "ConvBnRelu":     ConvBnRelu,
    "SEBlock":        SEBlock,
    "GLKA":           GLKA,
    "EfficientBlock": EfficientBlock,
}

__all__ = [
    "ConvBnRelu",
    "conv_bn_relu",
    "SEBlock",
    "GLKA",
    "EfficientBlock",
    "BLOCK_REGISTRY",
]
