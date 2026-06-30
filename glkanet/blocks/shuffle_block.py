from __future__ import annotations
import torch
import torch.nn as nn

try:
    from glkanet.blocks.conv_blocks import conv_bn_relu
    from glkanet.blocks.glka_block  import GLKA
except ImportError:
    from .conv_blocks import conv_bn_relu
    from .glka_block  import GLKA

# ──────────────────────────────────────────────────────────────
# ShuffleGLKABlock — Tree cấu trúc
# ──────────────────────────────────────────────────────────────
#
# ShuffleGLKABlock(in_channels, out_channels, stride, split_ratio)
# │
# ├── stride == 1   (basic unit, BẮT BUỘC in_channels == out_channels)
# │     input x (dim = in_channels) bị SPLIT làm 2 theo channel:
# │     │
# │     ├── x_id   (id_dim = in_channels - proc_dim)
# │     │     └── id_branch = nn.Identity()      (free, không tốn FLOPs)
# │     │
# │     └── x_proc (proc_dim = round(in_channels * split_ratio))
# │           └── proc_proj = nn.Identity() (proc_in == proc_out nên không cần match)
# │           └── proc:
# │                 ├── use_glka=True  → GLKA(dim=proc_dim, K=glka_K, stride=1,
# │                 │                          se_reduction=se_reduction)
# │                 └── use_glka=False → DWConv3x3(proc_dim, stride=1) + BN + ReLU6
# │
# │     out = concat([identity, proc], dim=1) → channel_shuffle(groups=2)
# │
# └── stride == 2   (downsample unit, in_channels có thể != out_channels)
#       input x (dim = in_channels) KHÔNG split — cả 2 nhánh đọc full x:
#       │
#       ├── id_branch (x_id = x, full in_channels)
#       │     └── DWConv3x3(in_channels, stride=2) + BN + ReLU6
#       │           → Conv1x1(in_channels → id_dim) + BN
#       │     (id_dim = out_channels - proc_dim, proc_dim = out_channels // 2)
#       │
#       └── proc nhánh (x_proc = x, full in_channels)
#             ├── proc_proj: Conv1x1(in_channels → proc_dim) + BN + ReLU6
#             │     (chỉ active nếu proc_in != proc_out, tức luôn active ở s=2)
#             └── proc:
#                   ├── use_glka=True  → GLKA(dim=proc_dim, K=glka_K, stride=2,
#                   │                          se_reduction=se_reduction)
#                   └── use_glka=False → Conv3x3(proc_dim, stride=2)/DWConv + BN + ReLU6
#
#       out = concat([identity, proc], dim=1) → channel_shuffle(groups=2)
#
# Forward (chung cho cả 2 case):
#   x_id, x_proc = split(x)            # s=1: chia theo channel | s=2: cả 2 = x
#   identity = id_branch(x_id)
#   proc     = proc(proc_proj(x_proc))
#   out      = concat([identity, proc])
#   return channel_shuffle(out, groups=2)
#
# Lưu ý: KHÔNG có residual add (+) như EfficientBlock — luồng thông tin
# truyền qua concat + shuffle, đúng tinh thần ShuffleNetV2 gốc.
#
# ──────────────────────────────────────────────────────────────


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(-1, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(-1, c, h, w)

class ShuffleGLKABlock(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int   = 1,
        split_ratio:  float = 0.5,
        use_glka:     bool  = True,
        glka_K:       int   = 13,
        se_reduction: int   = 0,
    ):
        super().__init__()
        self.stride       = stride
        self.in_channels  = in_channels
        self.out_channels = out_channels

        if stride == 1:
            assert in_channels == out_channels, (
                "ShuffleGLKABlock stride=1 yêu cầu in_channels == out_channels "
                "(giống basic unit ShuffleNetV2). Dùng stride=2 nếu cần đổi channel."
            )
            self.proc_dim = max(1, int(round(in_channels * split_ratio)))
            self.id_dim   = in_channels - self.proc_dim
            proc_in, proc_out = self.proc_dim, self.proc_dim
            id_in,   id_out   = self.id_dim,   self.id_dim
        else:
            # downsample unit: KHÔNG split input, cả 2 nhánh đọc full in_channels,
            # mỗi nhánh tự ra out_channels//2 (giống ShuffleNetV2 downsample unit)
            assert out_channels % 2 == 0, "out_channels phải chẵn khi stride=2"
            self.proc_dim = out_channels // 2
            self.id_dim   = out_channels - self.proc_dim
            proc_in, proc_out = in_channels, self.proc_dim
            id_in,   id_out   = in_channels, self.id_dim

        # ── Nhánh proc: GLKA hoặc DW thường ───────────────────────────
        if use_glka:
            self._proc_needs_proj = (proc_in != proc_out)
            if self._proc_needs_proj:
                # GLKA giữ nguyên channel -> cần 1x1 match channel trước
                self.proc_proj = nn.Sequential(
                    nn.Conv2d(proc_in, proc_out, 1, bias=False),
                    nn.BatchNorm2d(proc_out),
                    nn.ReLU6(inplace=True),
                )
            else:
                self.proc_proj = nn.Identity()
            self.proc = GLKA(
                dim          = proc_out,
                K            = glka_K,
                stride       = stride,
                se_reduction = se_reduction,
            )
        else:
            self._proc_needs_proj = False
            self.proc_proj = nn.Identity()
            self.proc = conv_bn_relu(
                proc_in, proc_out, kernel_size=3,
                stride=stride, padding=1,
                groups=1 if proc_in != proc_out else proc_out,
            )

        # ── Nhánh identity / downsample ────────────────────────────────
        if stride == 1:
            self.id_branch = nn.Identity()
        else:
            self.id_branch = nn.Sequential(
                nn.Conv2d(id_in, id_in, kernel_size=3,
                          stride=stride, padding=1,
                          groups=id_in, bias=False),
                nn.BatchNorm2d(id_in),
                nn.ReLU6(inplace=True),
                nn.Conv2d(id_in, id_out, 1, bias=False),
                nn.BatchNorm2d(id_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x_id, x_proc = x[:, :self.id_dim], x[:, self.id_dim:]
        else:
            x_id, x_proc = x, x  # downsample unit: cả 2 nhánh đọc full input

        identity = self.id_branch(x_id)

        proc = self.proc_proj(x_proc)
        proc = self.proc(proc)

        out = torch.cat([identity, proc], dim=1)
        return channel_shuffle(out, groups=2)

    def __repr__(self) -> str:
        proc_info = repr(self.proc) if isinstance(self.proc, GLKA) else "DWConv"
        return (
            f"ShuffleGLKABlock(in={self.in_channels}, out={self.out_channels}, "
            f"stride={self.stride}, proc_dim={self.proc_dim}, id_dim={self.id_dim}, "
            f"proc={proc_info})"
        )