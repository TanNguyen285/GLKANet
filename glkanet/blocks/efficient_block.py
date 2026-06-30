import torch
import torch.nn as nn

try:
    from glkanet.blocks.conv_blocks import conv_bn_relu
    from glkanet.blocks.glka_block  import GLKA
except ImportError:
    from .conv_blocks import conv_bn_relu
    from .glka_block  import GLKA

# ──────────────────────────────────────────────────────────────
# EfficientBlock — Tree cấu trúc
# ──────────────────────────────────────────────────────────────
#
# EfficientBlock(in_channels, out_channels, stride)
# │
# ├── shortcut (nhánh residual, chỉ active khi stride==1)
# │     ├── stride==1 & in==out  → nn.Identity()
# │     └── stride==1 & in!=out  → Conv1x1(in→out) + BN
# │     (stride==2 → không có shortcut, use_residual=False)
# │
# └── main path: x → expand → dw/glka → project → (+identity nếu residual)
#       │
#       ├── expand: Conv1x1(in_channels → hidden_dim) + BN + ReLU6
#       │     hidden_dim = in_channels * expansion_ratio
#       │
#       ├── dw / glka (rẽ nhánh theo use_glka, chỉ 1 trong 2 active)
#       │     │
#       │     ├── use_glka=True:
#       │     │     ├── dw   = nn.Identity()  (bỏ qua, GLKA tự lo stride)
#       │     │     └── glka = GLKA(dim=hidden_dim, K=glka_K, stride=stride,
#       │     │                     se_reduction=se_reduction)
#       │     │           │
#       │     │           ├── conv0: DW 5x5 (stride tại đây) + BN + ReLU6
#       │     │           ├── branches: multi dilated DWConv theo K-preset, sum lại
#       │     │           ├── SE gate (nếu se_reduction>0) hoặc Identity
#       │     │           └── output = anchor * (branch_out * gate)
#       │     │
#       │     └── use_glka=False:
#       │           ├── dw   = DWConv3x3(hidden_dim, stride=stride) + BN + ReLU6
#       │           └── glka = nn.Identity()
#       │
#       └── project: Conv1x1(hidden_dim → out_channels) + BN  (không activation)
#
# Forward:
#   identity = shortcut(x)
#   out = expand(x) → dw(out) → glka(out) → project(out)
#   return identity + out   if use_residual (stride==1)
#   return out              if stride==2
#
# ──────────────────────────────────────────────────────────────


class EfficientBlock(nn.Module):

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        stride:          int,
        expansion_ratio: int  = 2,
        use_glka:        bool = True,
        glka_K:          int  = 13,
        se_reduction:    int  = 0,
    ):
        super().__init__()
        self.stride = stride
        hidden_dim  = in_channels * expansion_ratio

        # s=1 → ép buộc residual dù in≠out (project 1×1 tự match channel)
        self.use_residual = (stride == 1)

        # ── Expand ────────────────────────────────────────────────────
        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)

        # ── Depthwise / GLKA ──────────────────────────────────────────
        if use_glka:
            # GLKA tự lo stride qua conv0 5×5 — không cần self.dw riêng
            self.dw   = nn.Identity()
            self.glka = GLKA(
                dim          = hidden_dim,
                K            = glka_K,
                stride       = stride,       # truyền stride xuống GLKA
                se_reduction = se_reduction,
            )
        else:
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=stride, padding=1, groups=hidden_dim)
            self.glka = nn.Identity()

        # ── Project ───────────────────────────────────────────────────
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── Shortcut: pointwise 1×1 nếu in≠out khi s=1 ───────────────
        if self.use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        out = self.project(out)
        if self.use_residual:
            return identity + out
        return out

    def __repr__(self) -> str:
        glka_info = repr(self.glka) if isinstance(self.glka, GLKA) else "off"
        return (
            f"EfficientBlock("
            f"residual={self.use_residual}, "
            f"stride={self.stride}, "
            f"glka={glka_info})"
        )