import torch.nn as nn


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    activation: bool = True,
) -> nn.Sequential:
    """Conv2d + BN (+ ReLU6 nếu activation=True)."""
    layers = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            groups=groups, bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class ConvBnRelu(nn.Sequential):
    """Module wrapper của conv_bn_relu — dùng được trong builder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = -1,          # -1 → tự tính same-padding
        groups: int = 1,
        activation: bool = True,
    ):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        super().__init__(
            *conv_bn_relu(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding,
                groups=groups, activation=activation,
            )
        )
