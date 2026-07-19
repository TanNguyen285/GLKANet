from __future__ import annotations
import torch
import torch.nn as nn

try:
    from glkanet.blocks.eca_block import ECABlock
except ImportError:
    from .eca_block import ECABlock

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


class GLKA_Shuffle(nn.Module):
    def __init__(
        self,
        dim:               int,
        out_channels:      int,
        K:                 int = 13,
        stride:            int = 1,
        attn_type:         str = "eca",
        use_residual:      bool = False,
        branches_config:   list | None = None,
        use_conv_replace:  bool = False,
        se_reduction:      int = 16,
    ):
        super().__init__()
        self.dim    = dim
        self.K      = K
        self.stride = stride
        self.use_conv_replace = use_conv_replace
        
        if stride == 1:
            self.attn_type = "none"
            self.use_residual = True
        else:
            self.attn_type = attn_type
            self.use_residual = use_residual

        self.use_split      = (stride == 1)
        self.branch_in_dim  = dim // 2 if self.use_split else dim
        self.mid_dim         = self.branch_in_dim  

        real_out_channels = 2 * self.branch_in_dim if self.use_split else self.branch_in_dim * 2
        
        if out_channels != real_out_channels:
            raise ValueError(
                f"GLKA_Shuffle: với dim={dim}, stride={stride} -> out_channels "
                f"phải là {real_out_channels}, nhưng yaml truyền out_channels={out_channels}."
            )
        self.out_channels = real_out_channels

        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(f"Không có preset cho K={K}")

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        self.pool_replace = nn.Conv2d(
            self.mid_dim, self.mid_dim, kernel_size=3, stride=stride,
            padding=1, groups=self.mid_dim, bias=False,
        )
        self.pool_replace_bn = nn.BatchNorm2d(self.mid_dim)

        assert self.attn_type in ("eca", "se", "none"), \
            f"attn_type phải là 'eca'/'se'/'none', nhận '{self.attn_type}'"
            
        if self.attn_type == "eca":
            self.eca = ECABlock(self.mid_dim)
        elif self.attn_type == "se":
            self.se = SEBlock(self.mid_dim, reduction=se_reduction)

        self.conv_nor = nn.Sequential(
            nn.Conv2d(self.branch_in_dim, self.branch_in_dim, kernel_size=3,
                      stride=stride, padding=1, groups=self.branch_in_dim, bias=False),
            nn.BatchNorm2d(self.branch_in_dim),
            nn.ReLU6(inplace=True),
        )

        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(self.mid_dim, self.mid_dim, k_size, padding=pad,
                          groups=self.mid_dim, dilation=dil, bias=False),
                nn.BatchNorm2d(self.mid_dim),
            ))

        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
        )

        self.reparam_conv: nn.Conv2d | None = None
        self._deployed = False

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        c = self.out_channels
        groups = 2
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.use_split:
            x_ca, x_sa = torch.chunk(x, chunks=2, dim=1)
        else:
            x_ca, x_sa = x, x

        if self.use_conv_replace:
            if self._deployed:
                ca = self.pool_replace(x_ca)
            else:
                ca = self.pool_replace_bn(self.pool_replace(x_ca))
        else:
            ca = self.maxpool(x_ca)

        if self.attn_type == "eca":
            ca = self.eca(ca)
        elif self.attn_type == "se":
            ca = self.se(ca)

        sa = self.conv_nor(x_sa)
        if self.reparam_conv is not None:
            sa = self.reparam_conv(sa)
        else:
            sa = sum(b(sa) for b in self.branches)
        sa = self.spatial_fuse(sa)

        out = torch.cat([ca, sa], dim=1)
        out = self._channel_shuffle(out)

        if self.stride == 1 and self.use_residual:
            out = out + identity

        return out

    def switch_to_deploy(self) -> None:
        if self._deployed:
            return

        if self.use_conv_replace:
            w_pr, b_pr = self._fuse_bn(nn.Sequential(self.pool_replace, self.pool_replace_bn))
            new_pool_replace = nn.Conv2d(
                self.mid_dim, self.mid_dim, 3, stride=self.stride,
                padding=1, groups=self.mid_dim, bias=True,
            )
            new_pool_replace.weight.data = w_pr
            new_pool_replace.bias.data   = b_pr
            self.pool_replace = new_pool_replace
            del self.pool_replace_bn
        else:
            if hasattr(self, "pool_replace"): del self.pool_replace
            if hasattr(self, "pool_replace_bn"): del self.pool_replace_bn

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

        w_nor, b_nor = self._fuse_bn(self.conv_nor[:2])
        new_conv_nor = nn.Conv2d(self.branch_in_dim, self.branch_in_dim, 3, stride=self.stride,
                                 padding=1, groups=self.branch_in_dim, bias=True)
        new_conv_nor.weight.data = w_nor
        new_conv_nor.bias.data   = b_nor
        self.conv_nor = nn.Sequential(new_conv_nor, nn.ReLU6(inplace=True))

        w_sf, b_sf = self._fuse_bn(self.spatial_fuse)
        new_spatial_fuse = nn.Conv2d(self.mid_dim, self.mid_dim, 1, bias=True)
        new_spatial_fuse.weight.data = w_sf
        new_spatial_fuse.bias.data   = b_sf
        self.spatial_fuse = new_spatial_fuse

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