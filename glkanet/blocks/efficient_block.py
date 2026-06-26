import torch
import torch.nn as nn

try:
    from glkanet.blocks.conv_blocks import conv_bn_relu
    from glkanet.blocks.glka_block  import GLKA
except ImportError:
    from .conv_blocks import conv_bn_relu
    from .glka_block  import GLKA


class EfficientBlock(nn.Module):
    """Inverted-residual block với tùy chọn GLKA thay depthwise conv.

    Args:
        in_channels:      channel đầu vào
        out_channels:     channel đầu ra
        stride:           1 (no downsample) hoặc 2 (downsample)
        expansion_ratio:  hệ số mở rộng channel ẩn
        use_glka:         True → dùng GLKA thay depthwise 3×3
        glka_K:           target kernel size của GLKA (mặc định 13)
        se_reduction:     tỉ lệ nén SE trong GLKA (mặc định 8)
    """

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        stride:          int  = 1,
        expansion_ratio: int  = 2,
        use_glka:        bool = False,
        glka_K:          int  = 13,
        se_reduction:    int  = 8,
    ):
        super().__init__()
        self.stride      = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = in_channels * expansion_ratio

        # Expand 1×1
        self.expand = nn.Sequential(
            *conv_bn_relu(in_channels, hidden, kernel_size=1)
        )

        # Depthwise / GLKA
        if use_glka:
            # Nếu stride=2 → cần downsample riêng trước GLKA
            self.dw = (
                nn.Sequential(*conv_bn_relu(
                    hidden, hidden, kernel_size=3,
                    stride=2, padding=1, groups=hidden,
                ))
                if stride == 2
                else nn.Identity()
            )
            self.glka = GLKA(hidden, K=glka_K, se_reduction=se_reduction)
        else:
            self.dw   = nn.Sequential(*conv_bn_relu(
                hidden, hidden, kernel_size=3,
                stride=stride, padding=1, groups=hidden,
            ))
            self.glka = nn.Identity()

        # Project 1×1 (no activation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        out = self.project(out)
        if self.use_residual:
            return identity + out
        return out
