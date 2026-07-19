from __future__ import annotations
import torch
import torch.nn as nn


class DepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, 1, pad,
                               groups=channels, bias=False)

    def forward(self, x):
        return self.conv(x)


class MaxPoolDownsample(nn.Module):
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(3, stride, 1)

    def forward(self, x):
        return self.pool(x)


class StridedConvDownsample(nn.Module):
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride, 1,
                               groups=channels, bias=False)

    def forward(self, x):
        return self.conv(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)
class ECA_Conv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Thay vì Conv1d, ta dùng Conv2d với kernel_size 1xK (hoặc Kx1)
        # Điều này giữ nguyên định dạng 4D (B, C, 1, 1), không cần transpose/squeeze nữa!
        self.conv = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1), 
                              padding=(kernel_size // 2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x có dạng [B, C, H, W]
        y = self.pool(x) # -> [B, C, 1, 1]
        
        # Đổi chiều nhẹ nhàng từ [B, C, 1, 1] sang [B, 1, C, 1] để quét Conv2D theo chiều dọc C
        y = y.permute(0, 2, 1, 3) 
        
        y = self.conv(y) # Quét dọc theo các channel
        y = self.sigmoid(y)
        
        # Đổi lại về [B, C, 1, 1] để nhân scale
        y = y.permute(0, 2, 1, 3) 
        
        return x * y

class Concat_Fusion(nn.Module):
    def __init__(self, channels: int, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1, groups=groups, bias=False)

    def forward(self, a, b):
        return self.conv(torch.cat([a, b], dim=1))


class ElementwiseFusion(nn.Module):
    def forward(self, a, b):
        return a * b


class DualInputWrapper(nn.Module):
    def __init__(self, fusion_module: nn.Module):
        super().__init__()
        self.fusion = fusion_module

    def forward(self, x):
        return self.fusion(x, x)


REGISTRY = {
    "dwconv3x3":    lambda ch: DepthwiseConv(ch, 3),
    "dwconv5x5":    lambda ch: DepthwiseConv(ch, 5),
    "dwconv7x7":    lambda ch: DepthwiseConv(ch, 7),
    "maxpool":      lambda ch: MaxPoolDownsample(ch),
    "stridedconv":  lambda ch: StridedConvDownsample(ch),
    "se":           lambda ch: SqueezeExcite(ch),
    "eca":          lambda ch: ECA(ch),
    "eca_conv2d":   lambda ch: ECA_Conv2d(ch),
    "concatg1":     lambda ch: DualInputWrapper(Concat_Fusion(ch, 1)),
    "concatg2":     lambda ch: DualInputWrapper(Concat_Fusion(ch, 2)),
    "concatg4":     lambda ch: DualInputWrapper(Concat_Fusion(ch, 4)),
    "elementwise":  lambda ch: DualInputWrapper(ElementwiseFusion()),
}
