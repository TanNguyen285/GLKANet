import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA-Net).

    Thay vì dùng 2 lớp FC (Conv 1x1) nén channel như SE, ECA dùng 1 Conv1D
    kernel size nhỏ trượt qua các channel (sau GAP) để học tương tác
    cục bộ giữa channel lân cận, tránh giảm chiều (dimensionality reduction)
    → giữ được thông tin, ít tham số hơn SE.

    Args:
        dim:        số channel đầu vào
        gamma, b:   hệ số dùng để tính kernel size thích ứng theo dim
                    k = |log2(dim)/gamma + b/gamma|, làm tròn về số lẻ gần nhất
    """

    def __init__(self, dim: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(dim) + b) / gamma))
        k = k if k % 2 else k + 1  # ép kernel size thành số lẻ
        k = max(k, 3)              # tối thiểu 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)