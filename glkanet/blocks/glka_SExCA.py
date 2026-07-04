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


class GLKA_SExCA(nn.Module):

    def __init__(
        self,
        dim:             int,
        K:               int = 13,
        stride:          int = 1,
        branches_config: list | None = None,
        se_reduction:    int = 8,      # bản gốc dùng //8, mặc định lại theo đó
    ):
        super().__init__()
        self.dim    = dim
        self.K      = K
        self.stride = stride

        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(
                f"Không có preset cho K={K}. Dùng K ∈ {list(GLKA_PRESETS)} "
                f"hoặc tự truyền branches_config=[[k,d],...]"
            )

        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5,
                      stride=stride, padding=2,
                      groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
        )

        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, k_size,
                          padding=pad, groups=dim, dilation=dil, bias=False),
                nn.BatchNorm2d(dim),
            ))

        # se_reduction <= 0 -> tắt hẳn SE (self.se = None). Không dùng
        # nn.Identity() vì forward sẽ nhân "anchor * self.se(anchor)",
        # với Identity thì thành anchor*anchor (bình phương sai công thức).
        self.se = SEBlock(dim, reduction=se_reduction) if se_reduction > 0 else None

        self.reparam_conv: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.conv0(x)

        # SEBlock đã tự nhân gating bên trong (x * sigmoid(...)), nên ở
        # đây chỉ gọi thẳng self.se(anchor), không nhân lại lần 2.
        anchor_gated = self.se(anchor) if self.se is not None else anchor

        if self.reparam_conv is not None:
            branch_sum = self.reparam_conv(anchor)
        else:
            branch_sum = sum(b(anchor) for b in self.branches)

        return anchor_gated * branch_sum

    def switch_to_deploy(self) -> None:
        if not hasattr(self, "branches"):
            return

        W_equiv = 0
        B_equiv = 0
        for branch, (k_size, dil) in zip(self.branches, self.branches_config):
            w_fused, b_fused = self._fuse_bn(branch)
            W_equiv += self._to_target_k(w_fused, orig_k=k_size, d=dil)
            B_equiv += b_fused

        self.reparam_conv = nn.Conv2d(
            self.dim, self.dim, self.K,
            padding=self.K // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = W_equiv
        self.reparam_conv.bias.data   = B_equiv

        del self.branches

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
        out    = torch.zeros((c, m, self.K, self.K),
                             device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out

    def __repr__(self) -> str:
        se_info = "off" if self.se is None else f"dim//{self.se.se[1].out_channels}"
        return (
            f"GLKA_SExCA(dim={self.dim}, K={self.K}, stride={self.stride}, "
            f"branches={self.branches_config}, SE={se_info})"
        )