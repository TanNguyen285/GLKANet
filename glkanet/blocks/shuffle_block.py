"""glkanet/blocks/shuffle_glka.py — Khối ShuffleNetV2 tích hợp Attention GLKA."""

from __future__ import annotations
import torch
import torch.nn as nn

try:
    from glkanet.blocks.conv_blocks import conv_bn_relu
    from glkanet.blocks.glka_block_base  import GLKA_CBAM
except ImportError:
    from .conv_blocks import conv_bn_relu
    from .glka_block_base  import GLKA_CBAM


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


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
        self.use_glka     = use_glka
        self._deployed    = False  # guard chống fold BN 2 lần khi GLKANet.switch_to_deploy() quét trùng

        if stride == 1:
            assert in_channels == out_channels, (
                "ShuffleGLKABlock stride=1 yêu cầu in_channels == out_channels. "
                "Dùng stride=2 nếu cần thay đổi channels."
            )
            self.proc_dim = max(1, int(round(in_channels * split_ratio)))
            self.id_dim   = in_channels - self.proc_dim
            proc_in, proc_out = self.proc_dim, self.proc_dim
            id_in,   id_out   = self.id_dim,   self.id_dim
        else:
            assert out_channels % 2 == 0, "out_channels phải chẵn khi stride=2"
            self.proc_dim = out_channels // 2
            self.id_dim   = out_channels - self.proc_dim
            proc_in, proc_out = in_channels, self.proc_dim
            id_in,   id_out   = in_channels, self.id_dim

        # ── Nhánh Xử lý (proc branch) ─────────────────────────────────
        if use_glka:
            self._proc_needs_proj = (proc_in != proc_out)
            if self._proc_needs_proj:
                self.proc_proj = nn.Sequential(
                    nn.Conv2d(proc_in, proc_out, 1, bias=False),
                    nn.BatchNorm2d(proc_out),
                    nn.ReLU6(inplace=True),
                )
            else:
                self.proc_proj = nn.Identity()

            self.proc = GLKA_CBAM(
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
                groups=proc_in if proc_in == proc_out else 1,
            )

        # ── Nhánh Giữ nguyên / Downsample định danh (identity branch) ──
        if stride == 1:
            self.id_branch = nn.Identity()
        else:
            self.id_branch = nn.Sequential(
                nn.Conv2d(id_in, id_in, kernel_size=3, stride=stride, padding=1, groups=id_in, bias=False),
                nn.BatchNorm2d(id_in),
                nn.ReLU6(inplace=True),
                nn.Conv2d(id_in, id_out, 1, bias=False),
                nn.BatchNorm2d(id_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x_id, x_proc = x[:, :self.id_dim], x[:, self.id_dim:]
        else:
            x_id, x_proc = x, x

        identity = self.id_branch(x_id)
        proc     = self.proc(self.proc_proj(x_proc))

        out = torch.cat([identity, proc], dim=1)
        return channel_shuffle(out, groups=2)

    def switch_to_deploy(self) -> None:
        """Reparam GLKA con + fold BN self-contained cho id_branch/proc_proj.
        Idempotent: gọi nhiều lần không sao vì có guard _deployed."""
        if self._deployed:
            return

        # 1) Reparam + fold BN bên trong GLKA con (idempotent sẵn ở phía GLKA)
        if self.use_glka and hasattr(self.proc, "switch_to_deploy"):
            self.proc.switch_to_deploy()

        # 2) Fold BN cho id_branch (chỉ tồn tại khi stride=2) — self-contained,
        #    an toàn vì mỗi BN fold vào đúng conv ngay trước nó.
        if self.stride != 1 and isinstance(self.id_branch, nn.Sequential):
            self.id_branch = self._fold_sequential_bn(
                self.id_branch,
                # (conv, bn) pairs: [0,1]=DW3x3+BN, [3,4]=PW1x1+BN ; index 2 là ReLU6 giữ nguyên
                pairs=[(0, 1), (3, 4)],
                relu_positions={2: nn.ReLU6(inplace=True)},
            )

        # 3) Fold BN cho proc_proj nếu có (1x1 conv + BN + ReLU6)
        if self._proc_needs_proj and isinstance(self.proc_proj, nn.Sequential):
            self.proc_proj = self._fold_sequential_bn(
                self.proc_proj,
                pairs=[(0, 1)],
                relu_positions={2: nn.ReLU6(inplace=True)},
            )

        self._deployed = True

    @staticmethod
    def _fold_sequential_bn(
        seq:           nn.Sequential,
        pairs:         list[tuple[int, int]],
        relu_positions: dict[int, nn.Module],
    ) -> nn.Sequential:
        """Fold từng cặp (conv_idx, bn_idx) trong 1 Sequential thành 1 conv có bias,
        giữ nguyên các lớp activation ở đúng vị trí cũ (không đổi thứ tự/logic)."""
        new_layers: dict[int, nn.Module] = {}
        max_idx = max(max(p) for p in pairs)
        if relu_positions:
            max_idx = max(max_idx, max(relu_positions.keys()))

        for conv_idx, bn_idx in pairs:
            conv: nn.Conv2d = seq[conv_idx]
            bn:   nn.BatchNorm2d = seq[bn_idx]
            std = (bn.running_var + bn.eps).sqrt()
            t   = (bn.weight / std).reshape(-1, 1, 1, 1)
            b_conv = conv.bias if conv.bias is not None else torch.zeros(
                conv.out_channels, device=conv.weight.device)
            w_fused = conv.weight * t
            b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)

            new_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True,
            )
            new_conv.weight.data = w_fused
            new_conv.bias.data   = b_fused
            new_layers[conv_idx] = new_conv

        for pos, act in relu_positions.items():
            new_layers[pos] = act

        ordered = [new_layers[i] for i in range(max_idx + 1) if i in new_layers]
        return nn.Sequential(*ordered)

    def __repr__(self) -> str:
        proc_info = repr(self.proc) if self.use_glka else "DWConv"
        return (
            f"ShuffleGLKABlock(in={self.in_channels}, out={self.out_channels}, "
            f"stride={self.stride}, proc_dim={self.proc_dim}, id_dim={self.id_dim}, "
            f"proc={proc_info})"
        )