from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════
# 1. ECABlock — Efficient Channel Attention
# ══════════════════════════════════════════════════════════════════════════

class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA-Net).
    Args:
        dim:      số channel đầu vào
        gamma, b: hệ số tính kernel size thích ứng theo dim
                  k = |log2(dim)/gamma + b/gamma|, làm tròn về số lẻ gần nhất
    """

    def __init__(self, dim: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(dim) + b) / gamma))
        k = k if k % 2 else k + 1   # ép kernel size thành số lẻ
        k = max(k, 3)                # tối thiểu 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                     # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)       # (B, 1, C)
        y = self.conv(y)                          # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)      # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ══════════════════════════════════════════════════════════════════════════
# 2. ConvBnRelu — Conv2d + BN (+ ReLU6)
# ══════════════════════════════════════════════════════════════════════════

def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    activation: bool = True,
) -> nn.Sequential:
    """Conv2d + BN (+ ReLU6 nếu activation=True)."""
    layers: list = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            groups=groups, bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class ConvBnRelu(nn.Sequential):
    """Module wrapper của conv_bn_relu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = -1,          # -1 -> tự tính same-padding
        groups: int = 1,
        activation: bool = True,
    ):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        super().__init__(
            *conv_bn_relu(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding,
                groups=groups, activation=activation,
            )
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. GLKA_Shuffle — khối attention 2 nhánh (channel-attn + large-kernel spatial)
# ══════════════════════════════════════════════════════════════════════════

GLKA_PRESETS: Dict[int, List[Tuple[int, int]]] = {
    13: [(3, 1), (3, 3), (5, 2), (5, 3), (13, 1)],
    7:  [(7, 1), (3, 2), (3, 3)],
    5:  [(5, 1), (3, 2)],
    3:  [(3, 1), (1, 1)],
}


class GLKA_Shuffle(nn.Module):
    """
    Nhánh trái (ca): MaxPool (hoặc depthwise-conv thay thế) -> ECA
    Nhánh phải (sa): depthwise 3x3 (Conv_Nor) -> multi-dilation depthwise
                     branches (Conv_spatial, tổng hợp theo K) -> fuse 1x1
    Ghép 2 nhánh theo channel rồi channel-shuffle.
    """

    def __init__(
        self,
        dim: int,
        out_channels: int,
        K: int = 13,
        stride: int = 1,
        branches_config: Optional[list] = None,
        use_conv_replace: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.stride = stride
        self.use_conv_replace = use_conv_replace

        self.use_split = (stride == 1)
        self.branch_in_dim = dim // 2 if self.use_split else dim
        self.mid_dim = self.branch_in_dim  # 2 nhánh luôn ra cùng số kênh, không proj

        real_out_channels = 2 * self.branch_in_dim
        if out_channels != real_out_channels:
            raise ValueError(
                f"GLKA_Shuffle: với dim={dim}, stride={stride} -> out_channels "
                f"phải là {real_out_channels} (thuần depthwise, không thêm proj "
                f"để ép kênh), nhưng truyền out_channels={out_channels}."
            )
        self.out_channels = real_out_channels

        if branches_config is not None:
            self.branches_config = [tuple(b) for b in branches_config]
        elif K in GLKA_PRESETS:
            self.branches_config = GLKA_PRESETS[K]
        else:
            raise ValueError(f"Không có preset cho K={K}")

        # ── Nhánh trái: MaxPool -> ECA (bản gốc)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        # ── Nhánh trái (bản thay thế): depthwise conv 3x3 cùng stride/padding
        self.pool_replace = nn.Conv2d(
            self.mid_dim, self.mid_dim, kernel_size=3, stride=stride,
            padding=1, groups=self.mid_dim, bias=False,
        )
        self.pool_replace_bn = nn.BatchNorm2d(self.mid_dim)

        self.eca = ECABlock(self.mid_dim)

        # ── Nhánh phải: Conv_Nor = depthwise 3x3 -> BN -> ReLU6
        self.conv_nor = nn.Sequential(
            nn.Conv2d(self.branch_in_dim, self.branch_in_dim, kernel_size=3,
                      stride=stride, padding=1, groups=self.branch_in_dim, bias=False),
            nn.BatchNorm2d(self.branch_in_dim),
            nn.ReLU6(inplace=True),
        )

        # ── Conv_spatial: multi-dilation depthwise branches, sum lại rồi fuse
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

        self.reparam_conv: Optional[nn.Conv2d] = None
        self._deployed = False

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        c = self.out_channels
        groups = 2
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_split:
            x_ca, x_sa = torch.chunk(x, chunks=2, dim=1)
        else:
            x_ca, x_sa = x, x

        # nhánh trái
        if self.use_conv_replace:
            if self._deployed:
                ca = self.pool_replace(x_ca)
            else:
                ca = self.pool_replace_bn(self.pool_replace(x_ca))
        else:
            ca = self.maxpool(x_ca)
        ca = self.eca(ca)

        # nhánh phải
        sa = self.conv_nor(x_sa)
        if self.reparam_conv is not None:
            sa = self.reparam_conv(sa)
        else:
            sa = sum(b(sa) for b in self.branches)
        sa = self.spatial_fuse(sa)

        out = torch.cat([ca, sa], dim=1)
        return self._channel_shuffle(out)

    # ── Deploy-time optimization (fuse BN, reparam multi-branch -> 1 conv KxK)
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
            new_pool_replace.bias.data = b_pr
            self.pool_replace = new_pool_replace
            del self.pool_replace_bn
        else:
            del self.pool_replace
            del self.pool_replace_bn

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
            self.reparam_conv.bias.data = B_equiv
            del self.branches

        w_nor, b_nor = self._fuse_bn(self.conv_nor[:2])
        new_conv_nor = nn.Conv2d(self.branch_in_dim, self.branch_in_dim, 3, stride=self.stride,
                                  padding=1, groups=self.branch_in_dim, bias=True)
        new_conv_nor.weight.data = w_nor
        new_conv_nor.bias.data = b_nor
        self.conv_nor = nn.Sequential(new_conv_nor, nn.ReLU6(inplace=True))

        w_sf, b_sf = self._fuse_bn(self.spatial_fuse)
        new_spatial_fuse = nn.Conv2d(self.mid_dim, self.mid_dim, 1, bias=True)
        new_spatial_fuse.weight.data = w_sf
        new_spatial_fuse.bias.data = b_sf
        self.spatial_fuse = new_spatial_fuse

        self._deployed = True

    def _fuse_bn(self, block: nn.Sequential):
        conv: nn.Conv2d = block[0]
        bn: nn.BatchNorm2d = block[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = (conv.bias if conv.bias is not None
                  else torch.zeros(conv.out_channels, device=conv.weight.device))
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


# ══════════════════════════════════════════════════════════════════════════
# 4. Dual_Attention_Block — wrapper mỏng quanh GLKA_Shuffle
# ══════════════════════════════════════════════════════════════════════════

class Dual_Attention_Block(nn.Module):
    """Wrapper mỏng quanh GLKA_Shuffle — không residual, không expand riêng.
    GLKA_Shuffle tự quyết định tách/không tách kênh dựa theo stride, và tự
    raise lỗi nếu out_channels truyền vào không khớp công thức bắt buộc."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        glka_K: int = 13,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self._deployed = False

        self.glka = GLKA_Shuffle(
            dim=in_channels,
            out_channels=out_channels,
            K=glka_K,
            stride=stride,
        )
        self.out_channels = self.glka.out_channels

    def forward(self, x):
        return self.glka(x)

    def switch_to_deploy(self) -> None:
        if self._deployed:
            return
        if hasattr(self.glka, "switch_to_deploy"):
            self.glka.switch_to_deploy()
        self._deployed = True

    def __repr__(self) -> str:
        return f"Dual_Attention_Block(stride={self.stride}, glka={repr(self.glka)})"


# ══════════════════════════════════════════════════════════════════════════
# 5. GLKABackbone — backbone đa tỉ lệ cho object detection
# ══════════════════════════════════════════════════════════════════════════

# Mỗi stage: (out_channels, stride_của_block_đầu_stage, glka_K)
# Block đầu mỗi stage downsample (stride=2), block thứ 2 giữ nguyên độ phân
# giải (stride=1) để tinh chỉnh đặc trưng — đúng cấu trúc trong yaml gốc.
StageSpec = Tuple[int, int, int]

STAGE_PRESETS: Dict[str, List[StageSpec]] = {
    # ⚠️ Sửa lại list K này cho khớp CHÍNH XÁC yaml bạn đang dùng nếu khác.
    "glka_large_kernel": [   # <-> runs/dual_ccmt (kernel 7x7/5x5 ở tầng giữa)
        (64,  2, 3),
        (128, 2, 5),
        (256, 2, 7),
        (512, 2, 3),
    ],
    "glka_full_3x3": [        # <-> runs/dual_ccmt_3x3 (full 3x3)
        (64,  2, 3),
        (128, 2, 3),
        (256, 2, 3),
        (512, 2, 3),
    ],
}

STEM_OUT_CHANNELS = 32


class GLKABackbone(nn.Module):
    """
    Backbone 4 stage (giống ResNet C2..C5) dùng Dual_Attention_Block, trả về
    dict feature map đa tỉ lệ — sẵn sàng cắm vào FPN cho object detection.

    stem:    224 -> 112                (stride 2,  32ch)
    stage1:  112 -> 56   -> "C2"       (stride 4,  64ch)
    stage2:  56  -> 28   -> "C3"       (stride 8,  128ch)
    stage3:  28  -> 14   -> "C4"       (stride 16, 256ch)
    stage4:  14  -> 7    -> "C5"       (stride 32, 512ch)
    """

    def __init__(
        self,
        stage_specs: Sequence[StageSpec] = STAGE_PRESETS["glka_large_kernel"],
        in_channels: int = 3,
        stem_out_channels: int = STEM_OUT_CHANNELS,
        return_stages: Sequence[str] = ("C2", "C3", "C4", "C5"),
    ):
        super().__init__()
        assert len(stage_specs) == 4, "Cần đúng 4 stage spec (C2, C3, C4, C5)."
        self.return_stages = list(return_stages)

        self.stem = ConvBnRelu(in_channels, stem_out_channels, kernel_size=3, stride=2)

        stage_names = ["C2", "C3", "C4", "C5"]
        self.stage_names = stage_names

        in_ch = stem_out_channels
        self.out_channels: Dict[str, int] = {}
        self.out_strides: Dict[str, int] = {}
        cur_stride = 2  # sau stem

        for name, (out_ch, stride, K) in zip(stage_names, stage_specs):
            block_down = Dual_Attention_Block(in_ch, out_ch, stride=stride, glka_K=K)
            block_refine = Dual_Attention_Block(out_ch, out_ch, stride=1, glka_K=K)
            setattr(self, f"stage_{name}", nn.Sequential(block_down, block_refine))

            cur_stride *= stride
            self.out_channels[name] = out_ch
            self.out_strides[name] = cur_stride
            in_ch = out_ch

        # Tổng số params & thông tin nhanh
        self._built_from = stage_specs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.stage_names:
            stage = getattr(self, f"stage_{name}")
            x = stage(x)
            if name in self.return_stages:
                outputs[name] = x
        return outputs

    def switch_to_deploy(self) -> "GLKABackbone":
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        return self

    def info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print(f"[GLKABackbone] {total/1e6:.3f}M params")
        for name in self.stage_names:
            marker = "*" if name in self.return_stages else " "
            print(f"  [{marker}] {name}: {self.out_channels[name]}ch, "
                  f"stride {self.out_strides[name]}")


# ══════════════════════════════════════════════════════════════════════════
# 6. Builder tiện dụng + load pretrained (từ checkpoint classification)
# ══════════════════════════════════════════════════════════════════════════

def build_glka_backbone(
    variant: str = "glka_large_kernel",
    pretrained_path: Optional[str] = None,
    return_stages: Sequence[str] = ("C2", "C3", "C4", "C5"),
    strict_load: bool = False,
) -> GLKABackbone:
    """
    variant: "glka_large_kernel" | "glka_full_3x3" | custom list stage_specs
    pretrained_path: đường dẫn checkpoint classification gốc (state_dict,
        không phải bản deploy/TorchScript) -> tự lọc và chỉ load các tensor
        thuộc phần backbone_layers (bỏ head.*).
    """
    if variant not in STAGE_PRESETS:
        raise ValueError(f"variant '{variant}' không có trong STAGE_PRESETS: {list(STAGE_PRESETS.keys())}")

    backbone = GLKABackbone(stage_specs=STAGE_PRESETS[variant], return_stages=return_stages)

    if pretrained_path:
        _load_pretrained_backbone(backbone, pretrained_path, strict=strict_load)

    return backbone


def _load_pretrained_backbone(backbone: GLKABackbone, ckpt_path: str, strict: bool = False) -> None:

    """Mapping index -> tên trong backbone mới:
        backbone_layers.0            -> stem
        backbone_layers.1            -> stage_C2.0
        backbone_layers.2            -> stage_C2.1
        backbone_layers.3            -> stage_C3.0
        backbone_layers.4            -> stage_C3.1
        backbone_layers.5            -> stage_C4.0
        backbone_layers.6            -> stage_C4.1
        backbone_layers.7            -> stage_C5.0
        backbone_layers.8            -> stage_C5.1
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]):
            state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(
            f"Không nhận diện được state_dict trong {ckpt_path}. "
            f"Keys top-level: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    # index cũ -> tên mới
    old_to_new_prefix = {
        0: "stem",
        1: "stage_C2.0", 2: "stage_C2.1",
        3: "stage_C3.0", 4: "stage_C3.1",
        5: "stage_C4.0", 6: "stage_C4.1",
        7: "stage_C5.0", 8: "stage_C5.1",
    }

    remapped = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone_layers."):
            continue  # bỏ qua toàn bộ head.*
        parts = k.split(".")
        idx = int(parts[1])
        rest = ".".join(parts[2:])
        if idx not in old_to_new_prefix:
            continue
        new_key = f"{old_to_new_prefix[idx]}.{rest}"
        remapped[new_key] = v

    missing, unexpected = backbone.load_state_dict(remapped, strict=strict)
    n_loaded = len(remapped) - len(unexpected)
    print(f"[build_glka_backbone] Loaded {n_loaded}/{len(remapped)} tensors từ '{ckpt_path}'")
    if missing:
        print(f"  [!] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  [!] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


# ══════════════════════════════════════════════════════════════════════════
# 7. Quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for variant in ("glka_large_kernel", "glka_full_3x3"):
        print(f"\n=== {variant} ===")
        backbone = build_glka_backbone(variant)
        backbone.info()
        x = torch.randn(1, 3, 224, 224)
        feats = backbone(x)
        for name, f in feats.items():
            print(f"  {name}: {tuple(f.shape)}")