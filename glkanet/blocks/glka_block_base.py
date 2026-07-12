from __future__ import annotations
import torch
import torch.nn as nn

GLKA_PRESETS: dict[int, list[tuple[int, int]]] = {
    13: [(3, 1), (3, 3), (5, 2), (5, 3), (13, 1)],
    7:  [(7, 1), (3, 2), (3, 3)],
    5:  [(5, 1), (3, 2)],
    3:  [(3, 1), (1, 1)],
}


class GLKA_CBAM(nn.Module):
    """
    Conv_Nor (KxK, DW) -> anchor
    anchor * SE(anchor) -> anchor_se     (SE bị tắt hẳn nếu se_reduction=0 -> không
                                           tạo self.se, không tốn compute/param nào)
    anchor_se -> Spatial (sum các nhánh dilated, tổng hợp field KxK) -> sigmoid -> spatial_gate
    anchor_se * spatial_gate -> output

    se_reduction=0  -> KHÔNG dùng SE, anchor_se = anchor (bỏ hẳn nhánh channel gate)
    se_reduction>0  -> dùng SE với hidden = max(1, dim // se_reduction)
    """
    def __init__(
        self,
        dim:             int,
        K:               int = 13,
        stride:          int = 1,
        conv0_k:         int = 5,
        branches_config: list | None = None,
        se_reduction:    int = 4,
    ):
        super().__init__()
        self.dim     = dim
        self.K       = K
        self.stride  = stride
        self.conv0_k = conv0_k

        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(f"Không có preset cho K={K}. Chọn K ∈ {list(GLKA_PRESETS)} hoặc tự truyền config.")

        # 1) Conv_Nor: DW Conv KxK + BN + ReLU -> tạo anchor
        pad0 = conv0_k // 2
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=conv0_k, stride=stride, padding=pad0, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
        )

        # 2) SE (Channel Attention) — áp dụng SAU conv0, lên chính anchor.
        #    se_reduction=0 -> KHÔNG tạo self.se, forward() tự bỏ qua nhánh này
        #    (dùng hasattr thay vì cờ bool riêng -> không còn 2 nguồn sự thật).
        self.use_se = se_reduction > 0
        self.last_gate = None  # cache (B,C,1,1) cho loss bank đọc lại nếu cần

        if self.use_se:
            hidden = max(1, dim // se_reduction)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, dim, 1, bias=True),
                nn.Sigmoid(),
            )

        # 3) Spatial: các nhánh Dilated (tổng hợp thành field lớn KxK), input là anchor_se
        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, k_size, padding=pad, groups=dim, dilation=dil, bias=False),
                nn.BatchNorm2d(dim),
            ))
        self.spatial_gate_act = nn.Sigmoid()

        self.reparam_conv: nn.Conv2d | None = None
        self._deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bước 1: Conv_Nor -> anchor
        anchor = self.conv0(x)

        # Bước 2: SE nhân lên anchor (bỏ qua nếu se_reduction=0 hoặc đã strip)
        if self.use_se and hasattr(self, "se"):
            self.last_gate = self.se(anchor)
            anchor_se = anchor * self.last_gate
        else:
            self.last_gate = None
            anchor_se = anchor

        # Bước 3: Spatial attention từ anchor_se
        if self.reparam_conv is not None:
            spatial_gate = self.reparam_conv(anchor_se)
        else:
            spatial_gate = sum(branch(anchor_se) for branch in self.branches)
        spatial_gate = self.spatial_gate_act(spatial_gate)

        # Bước 4: Nhân lần cuối
        return anchor_se * spatial_gate

    def strip_se(self) -> None:
        """Xoá hẳn nhánh SE — gọi lúc deploy, sau khi train xong (tiết kiệm compute vĩnh viễn)."""
        self.use_se = False
        if hasattr(self, "se"):
            del self.se

    def switch_to_deploy(self) -> None:
        if self._deployed:
            return

        # 1) Reparam các nhánh dilated -> 1 conv KxK duy nhất (Spatial Attention)
        if hasattr(self, "branches"):
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

        # 2) Fold BN vào chính conv0
        w0, b0 = self._fuse_bn(self.conv0[:2])
        new_conv0 = nn.Conv2d(self.dim, self.dim, self.conv0_k, stride=self.stride,
                               padding=self.conv0_k // 2, groups=self.dim, bias=True)
        new_conv0.weight.data = w0
        new_conv0.bias.data   = b0
        self.conv0 = nn.Sequential(new_conv0, nn.ReLU6(inplace=True))

        self._deployed = True

    def _fuse_bn(self, block: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        conv: nn.Conv2d = block[0]
        bn: nn.BatchNorm2d = block[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)

        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
        return w_fused, b_fused

    def _to_target_k(self, kernel: torch.Tensor, orig_k: int, d: int) -> torch.Tensor:
        c, m = kernel.shape[:2]
        kd = (orig_k - 1) * d + 1
        out = torch.zeros((c, m, self.K, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out

    def __repr__(self) -> str:
        return (
            f"GLKA_CBAM(dim={self.dim}, conv0_k={self.conv0_k}, K={self.K}, "
            f"stride={self.stride}, branches={self.branches_config}, use_se={self.use_se})"
        )