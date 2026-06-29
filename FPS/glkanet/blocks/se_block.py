import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention gate.

    Args:
        dim:        số channel đầu vào
        reduction:  tỉ lệ nén (mặc định 8 → dim//8 hidden units)
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)
