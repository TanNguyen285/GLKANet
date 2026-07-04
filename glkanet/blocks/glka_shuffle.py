from __future__ import annotations
import torch
import torch.nn as nn

try:
    from glkanet.blocks.se_block import SEBlock
except ImportError:
    from .se_block import SEBlock

GLKA_PRESETS: dict[int, list[tuple[int, int]]] = {
    13: [(3, 1), (3, 3), (5, 2), (5, 3), (13, 1)],
    7:  [(7, 1), (3, 2), (3, 3)],
    5:  [(5, 1), (3, 2)],
    3:  [(3, 1), (1, 1)],
}


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    # Dùng -1 cho batch (suy ra tự động, không cần đọc x.shape[0] tường minh) và ép
    # c, h, w về Python int cố định khi có thể — giảm mạnh số node Shape/Gather/Cast/
    # Unsqueeze thừa trong đồ thị ONNX (mỗi lần gọi hàm này vốn sinh ~10 node phụ trợ
    # chỉ để tính lại kích thước động lúc runtime, dù kết quả luôn giống nhau).
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    x = x.reshape(-1, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(-1, c, h, w)


class GLKA_Shuffle(nn.Module):
    def __init__(
        self,
        dim:             int,
        out_channels:    int,
        K:               int = 13,
        stride:          int = 1,
        branches_config: list | None = None,
        se_reduction:    int = 8,
    ):
        super().__init__()
        self.dim          = dim
        self.out_channels = out_channels
        self.K            = K
        self.stride       = stride
        self.mid_dim      = dim // 2

        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(f"Không có preset cho K={K}")

        # Tầng Conv_local DUY NHẤT trước khi rẽ nhánh
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=stride, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
        )

        # Nhánh trái: Channel Attention (SE) — nhận thẳng x_left
        self.se = SEBlock(self.mid_dim, reduction=se_reduction) if se_reduction > 0 else None

        # Nhánh phải: Spatial Attention — nhận thẳng x_right
        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(self.mid_dim, self.mid_dim, k_size, padding=pad, groups=self.mid_dim, dilation=dil, bias=False),
                nn.BatchNorm2d(self.mid_dim),
            ))

        # Fusion: conv1x1 groups=2, trộn thông tin nhờ channel_shuffle chạy trước đó
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, out_channels, kernel_size=1, groups=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.reparam_conv: nn.Conv2d | None = None
        self._deployed = False  # True sau switch_to_deploy(): bỏ shuffle runtime, BN đã fold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.conv0(x)
        x_left, x_right = torch.chunk(anchor, chunks=2, dim=1)

        ca = self.se(x_left) if self.se is not None else x_left

        if self.reparam_conv is not None:
            sa = self.reparam_conv(x_right)
        else:
            sa = sum(b(x_right) for b in self.branches)

        out = torch.cat([ca, sa], dim=1)
        out = channel_shuffle(out, groups=2)  # luôn chạy, kể cả deploy — không thể fold qua groups=2 conv
        return self.fuse(out)

    # ── Deploy-time optimization ─────────────────────────────────────
    def switch_to_deploy(self) -> None:
        if self._deployed:
            return

        # 1) Reparam 4 nhánh dilated -> 1 conv KxK
        if hasattr(self, "branches"):
            W_equiv = 0
            B_equiv = 0
            for branch, (k_size, dil) in zip(self.branches, self.branches_config):
                w_fused, b_fused = self._fuse_bn(branch)
                W_equiv += self._to_target_k(w_fused, orig_k=k_size, d=dil)
                B_equiv += b_fused

            self.reparam_conv = nn.Conv2d(
                self.mid_dim, self.mid_dim, self.K, padding=self.K // 2,
                groups=self.mid_dim, bias=True,
            )
            self.reparam_conv.weight.data = W_equiv
            self.reparam_conv.bias.data   = B_equiv
            del self.branches

        # 2) Fold BN vào chính conv0 (bias-add, không cần BN runtime nữa)
        w0, b0 = self._fuse_bn(self.conv0[:2])
        new_conv0 = nn.Conv2d(self.dim, self.dim, 5, stride=self.stride,
                               padding=2, groups=self.dim, bias=True)
        new_conv0.weight.data = w0
        new_conv0.bias.data   = b0
        self.conv0 = nn.Sequential(new_conv0, nn.ReLU6(inplace=True))

        # 3) KHÔNG fold channel_shuffle vào fuse — fuse là Conv2d(groups=2), weight
        #    chỉ có shape (out, dim/groups, 1, 1). Shuffle trộn thông tin XUYÊN 2 group,
        #    còn conv groups=2 chỉ nhìn được trong phạm vi group của chính nó -> không có
        #    cách nào biểu diễn phép trộn xuyên-group bằng cách permute weight của 1 conv
        #    bị giới hạn trong group. Bản fold trước đó SAI (đã gây MISMATCH lúc export).
        #    -> channel_shuffle() vẫn chạy runtime bình thường, kể cả ở deploy mode.

        # 4) Fold BN vào chính fuse conv (vẫn hợp lệ, không liên quan tới shuffle)
        wf, bf = self._fuse_bn(self.fuse)
        new_fuse = nn.Conv2d(self.dim, self.out_channels, 1, groups=2, bias=True)
        new_fuse.weight.data = wf
        new_fuse.bias.data   = bf
        self.fuse = new_fuse

        self._deployed = True

    def _fuse_bn(self, block: nn.Sequential):
        conv: nn.Conv2d      = block[0]
        bn:   nn.BatchNorm2d = block[1]
        std = (bn.running_var + bn.eps).sqrt()
        t   = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv  = (conv.bias if conv.bias is not None
                   else torch.zeros(conv.out_channels, device=conv.weight.device))
        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
        return w_fused, b_fused

    def _to_target_k(self, kernel: torch.Tensor, orig_k: int, d: int) -> torch.Tensor:
        c, m   = kernel.shape[:2]
        kd     = (orig_k - 1) * d + 1
        out    = torch.zeros((c, m, self.K, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out