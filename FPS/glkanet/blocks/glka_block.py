from __future__ import annotations
import torch
import torch.nn as nn
# ──────────────────────────────────────────────────────────────
# Branches preset theo K
# ──────────────────────────────────────────────────────────────

GLKA_PRESETS: dict[int, list[tuple[int, int]]] = {
    13: [(3, 1), (3, 3), (5, 2), (5, 3), (13, 1)],
    7:  [(7, 1), (3, 2), (3, 3)],
    5:  [(5, 1), (3, 2)],
    3:  [(3, 1), (1, 1)],
}
# ──────────────────────────────────────────────────────────────
# GLKA Block
# ──────────────────────────────────────────────────────────────

class GLKA(nn.Module):
    def __init__(
        self,
        dim:             int,
        K:               int = 13,
        stride:          int = 1,
        branches_config: list | None = None,
        se_reduction:    int = 0,
    ):
        super().__init__()
        self.dim    = dim
        self.K      = K
        self.stride = stride

        # ── Branches config ──────────────────────────────────────────
        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(
                f"Không có preset cho K={K}. "
                f"Dùng K ∈ {list(GLKA_PRESETS)} "
                f"hoặc tự truyền branches_config=[[k,d],...]"
            )

        # ── Conv0: DW 5×5, đảm nhiệm stride + trích đặc trưng global ──
        # stride=2 → downsample luôn tại đây, EB không cần DW 3×3 riêng
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5,
                      stride=stride, padding=2,
                      groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
        )

        # ── Dilated branches (chạy trên output conv0, không stride) ──
        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, k_size,
                          padding=pad, groups=dim, dilation=dil, bias=False),
                nn.BatchNorm2d(dim),
            ))

        # ── SE gate — chạy SAU sum branches ─────────────────────────
        if se_reduction > 0:
            hidden = max(1, dim // se_reduction)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, dim, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.se = nn.Identity()

        # Slot reparam (None khi train)
        self.reparam_conv: nn.Conv2d | None = None

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv0: 5×5 DW, stride, BN, ReLU6 → anchor
        anchor = self.conv0(x)
        # branches chạy trên anchor
        if self.reparam_conv is not None:
            branch_out = self.reparam_conv(anchor)
        else:
            branch_out = sum(b(anchor) for b in self.branches)

        # SE chạy sau sum → channel gate
        gate = self.se(branch_out)          # Sigmoid hoặc Identity

        return anchor * (branch_out * gate)
    # ── Structural reparameterization ─────────────────────────────────
    def switch_to_deploy(self) -> None:
        if not hasattr(self, "branches"):
            return  # đã deploy rồi

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

    # ── Helpers ───────────────────────────────────────────────────────

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

    def _to_target_k(
        self,
        kernel: torch.Tensor,
        orig_k: int,
        d:      int,
    ) -> torch.Tensor:
        """Đặt kernel dilated vào grid K×K, center-aligned."""
        c, m   = kernel.shape[:2]
        kd     = (orig_k - 1) * d + 1
        out    = torch.zeros((c, m, self.K, self.K),
                             device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out

    def __repr__(self) -> str:
        se_info = (
            "off" if isinstance(self.se, nn.Identity)
            else f"dim//{self.se[1].out_channels}"
        )
        return (
            f"GLKA(dim={self.dim}, K={self.K}, stride={self.stride}, "
            f"branches={self.branches_config}, SE={se_info})"
        )
# ──────────────────────────────────────────────────────────────
# Helpers dùng chung
# ──────────────────────────────────────────────────────────────

def conv_bn_relu(
    in_channels:  int,
    out_channels: int,
    kernel_size:  int,
    stride:       int = 1,
    padding:      int = 0,
    groups:       int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


# ──────────────────────────────────────────────────────────────
# EfficientBlock
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

        # s=1 → ép buộc residual dù in≠out
        self.use_residual = (stride == 1)

        # ── Expand ──────────────────────────────────────────────────
        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)

        # ── Depthwise / GLKA ────────────────────────────────────────
        if use_glka:
            # GLKA.conv0 5×5 tự lo stride → không cần self.dw
            self.glka = GLKA(
                dim          = hidden_dim,
                K            = glka_K,
                stride       = stride,
                se_reduction = se_reduction,
            )
        else:
            # Không dùng GLKA → DW conv thường giữ stride
            self.glka = nn.Sequential(
                conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                             stride=stride, padding=1, groups=hidden_dim)
            )

        # ── Project ─────────────────────────────────────────────────
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
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

# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import copy

    print("=== Test GLKA presets ===")
    for K in [13, 7, 5, 3]:
        g = GLKA(dim=64, K=K, stride=1, se_reduction=8)
        print(f"  K={K}: {g}")

    print("\n=== Test EfficientBlock s=2 (GLKA tự stride) ===")
    eb = EfficientBlock(32, 64, stride=2, use_glka=True, glka_K=13, se_reduction=8)
    x2 = torch.randn(1, 32, 56, 56)
    out2 = eb(x2)
    print(f"  in={x2.shape} → out={out2.shape}")   # expect [1,64,28,28]

    print("\n=== Test EfficientBlock s=1, in==out (Identity shortcut) ===")
    eb_same = EfficientBlock(32, 32, stride=1, use_glka=True, glka_K=7, se_reduction=0)
    x3 = torch.randn(1, 32, 28, 28)
    out3 = eb_same(x3)
    print(f"  in={x3.shape} → out={out3.shape}")   # expect [1,32,28,28]

    print("\n=== Test EfficientBlock s=1, in≠out (Conv1×1 shortcut) ===")
    eb_diff = EfficientBlock(32, 64, stride=1, use_glka=True, glka_K=5, se_reduction=4)
    x4 = torch.randn(1, 32, 28, 28)
    out4 = eb_diff(x4)
    print(f"  in={x4.shape} → out={out4.shape}")   # expect [1,64,28,28]

    print("\n=== Test reparam: train vs deploy ===")
    model = EfficientBlock(32, 32, stride=1, use_glka=True, glka_K=13, se_reduction=0)
    model.eval()
    x = torch.randn(1, 32, 56, 56)

    with torch.no_grad():
        out_train = model(x)

    model_d = copy.deepcopy(model)
    model_d.glka.switch_to_deploy()

    with torch.no_grad():
        out_deploy = model_d(x)

    diff = (out_train - out_deploy).abs().max().item()
    print(f"  Max diff train vs deploy: {diff:.2e}")
    assert diff < 1e-4, "Reparam sai!"
    print("  OK")

    print("\n=== Test reparam: s=2, train vs deploy ===")
    model2 = EfficientBlock(32, 64, stride=2, use_glka=True, glka_K=13, se_reduction=8)
    model2.eval()
    x5 = torch.randn(1, 32, 56, 56)

    with torch.no_grad():
        out_train2 = model2(x5)

    model2_d = copy.deepcopy(model2)
    model2_d.glka.switch_to_deploy()

    with torch.no_grad():
        out_deploy2 = model2_d(x5)

    diff2 = (out_train2 - out_deploy2).abs().max().item()
    print(f"  Max diff train vs deploy (s=2): {diff2:.2e}")
    assert diff2 < 1e-4, "Reparam s=2 sai!"
    print("  OK")