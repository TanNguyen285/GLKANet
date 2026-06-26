import torch
import torch.nn as nn

try:
    from glkanet.blocks.se_block import SEBlock
except ImportError:
    from .se_block import SEBlock


class GLKA(nn.Module):
    def __init__(self, dim: int, K: int = 13, se_reduction: int = 8):
        super().__init__()
        self.dim = dim
        self.K = K

        # Conv 5×5 depthwise — tạo shared feature map cho tất cả branch
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # SE gate trên global_conv
        self.se = SEBlock(dim, reduction=se_reduction)

        # 4 dilated branches (mỗi branch: depthwise conv + BN)
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1,  groups=dim, dilation=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=3,  groups=dim, dilation=3, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=4,  groups=dim, dilation=2, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=6,  groups=dim, dilation=3, bias=False),
            nn.BatchNorm2d(dim),
        )

        # Slot cho reparam conv (None khi đang train)
        self.reparam_conv: nn.Conv2d | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_conv = self.conv0(x)
        anchor = self.se(global_conv)          # SE gate

        if self.reparam_conv is not None:
            branch_out = self.reparam_conv(global_conv)
        else:
            branch_out = (
                self.branch1(global_conv)
                + self.branch2(global_conv)
                + self.branch3(global_conv)
                + self.branch4(global_conv)
            )

        return anchor * branch_out

    # ------------------------------------------------------------------
    # Structural reparameterization
    # ------------------------------------------------------------------
    def switch_to_deploy(self) -> None:
        if self.reparam_conv is not None:
            return  # đã deploy rồi

        w1, b1 = self._fuse_bn(self.branch1)
        w2, b2 = self._fuse_bn(self.branch2)
        w3, b3 = self._fuse_bn(self.branch3)
        w4, b4 = self._fuse_bn(self.branch4)

        W = (
            self._dilate_to_target(w1, orig_k=3, d=1)
            + self._dilate_to_target(w2, orig_k=3, d=3)
            + self._dilate_to_target(w3, orig_k=5, d=2)
            + self._dilate_to_target(w4, orig_k=5, d=3)
        )
        B = b1 + b2 + b3 + b4

        self.reparam_conv = nn.Conv2d(
            self.dim, self.dim, self.K,
            padding=self.K // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data.copy_(W)
        self.reparam_conv.bias.data.copy_(B)

        # Xóa branch để giảm memory
        del self.branch1, self.branch2, self.branch3, self.branch4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fuse_bn(self, block: nn.Sequential):
        """Trả về (w_fused, b_fused) sau khi fold BN vào Conv."""
        conv: nn.Conv2d   = block[0]
        bn:   nn.BatchNorm2d = block[1]

        std = (bn.running_var + bn.eps).sqrt()
        t   = (bn.weight / std).reshape(-1, 1, 1, 1)

        b_conv  = (conv.bias if conv.bias is not None
                   else torch.zeros(conv.out_channels, device=conv.weight.device))
        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
        return w_fused, b_fused

    def _dilate_to_target(
        self,
        kernel: torch.Tensor,
        orig_k: int,
        d: int,
    ) -> torch.Tensor:
        c, m = kernel.shape[:2]
        kd = (orig_k - 1) * d + 1          # effective kernel size

        out    = torch.zeros((c, m, self.K, self.K),
                             device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out
