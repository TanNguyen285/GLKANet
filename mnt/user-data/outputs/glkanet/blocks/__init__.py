"""Block registry — mọi block đều đăng ký ở đây để builder.py import."""

from glkanet.blocks.conv_blocks     import ConvBnRelu, conv_bn_relu
from glkanet.blocks.se_block        import SEBlock
from glkanet.blocks.glka_block      import GLKA
from glkanet.blocks.efficient_block import EfficientBlock


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
