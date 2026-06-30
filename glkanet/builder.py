"""glkanet/builder.py — đọc yaml config → build GLKANet động."""

from __future__ import annotations

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Any

try:
    from glkanet.blocks import BLOCK_REGISTRY, ConvBnRelu, EfficientBlock, ShuffleGLKABlock
    from glkanet.blocks import GLKA as _GLKABlock
except ImportError:
    from blocks import BLOCK_REGISTRY, ConvBnRelu, EfficientBlock, ShuffleGLKABlock
    from blocks import GLKA as _GLKABlock


# ──────────────────────────────────────────────────────────────
# Classifier head
# ──────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    GAP → Flatten → [BN1d] → [Linear(in→mid) + BN1d + Hardswish] → Dropout → Linear(→nc)

    Args:
        in_features:  số channels từ backbone (sau GAP+Flatten)
        num_classes:  số class đầu ra
        dropout:      tỉ lệ dropout trước FC cuối (default 0.2)
        mid_features: nếu > 0, thêm bottleneck Linear(in→mid) trước FC chính
                      giúp học feature tốt hơn khi backbone out lớn (vd 512→256→nc)
                      nếu = 0, tắt bottleneck
        use_bn:       thêm BN1d ngay sau Flatten (trước bottleneck hoặc Dropout)
                      giúp normalize features sau GAP, thường tăng ~0.5–1% acc
    """
    def __init__(
        self,
        in_features:  int,
        num_classes:  int,
        dropout:      float = 0.2,
        mid_features: int   = 0,
        use_bn:       bool  = False,
    ):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []

        # BN1d ngay sau Flatten (optional)
        if use_bn:
            layers.append(nn.BatchNorm1d(in_features))

        if mid_features > 0:
            # Bottleneck: in → mid → nc
            layers += [
                nn.Linear(in_features, mid_features, bias=False),
                nn.BatchNorm1d(mid_features),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(mid_features, num_classes),
            ]
        else:
            # Bare: in → nc
            layers += [
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            ]

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x        = self.pool(x)
        features = self.flatten(x)
        logits   = self.classifier(features)
        return logits, features


# ──────────────────────────────────────────────────────────────
# GLKANet wrapper
# ──────────────────────────────────────────────────────────────

class GLKANet(nn.Module):
    def __init__(
        self,
        backbone_layers: nn.ModuleList,
        head: ClassifierHead,
        layer_channels: list[int],
    ):
        super().__init__()
        self.backbone_layers = backbone_layers
        self.head            = head
        self.layer_channels  = layer_channels

    def forward(self, x: torch.Tensor):
        for layer in self.backbone_layers:
            x = layer(x)
        return self.head(x)

    def switch_to_deploy(self) -> "GLKANet":
        for m in self.modules():
            if isinstance(m, _GLKABlock):
                m.switch_to_deploy()
        return self

    def is_deployed(self) -> bool:
        glka_modules = [m for m in self.modules() if isinstance(m, _GLKABlock)]
        return bool(glka_modules) and all(
            m.reparam_conv is not None for m in glka_modules
        )

    def info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print(f"[GLKANet] {total/1e6:.3f}M params | deployed={self.is_deployed()}")
        for i, (ch, layer) in enumerate(
            zip(self.layer_channels, self.backbone_layers)
        ):
            print(f"  [{i:02d}] {layer.__class__.__name__:<20} → {ch}ch")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _out_channels_of(block: nn.Module, in_ch: int, args: list) -> int:
    if isinstance(block, (ConvBnRelu, EfficientBlock, ShuffleGLKABlock)):
        return args[0]
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, 32, 32)
        out   = block(dummy)
        if isinstance(out, tuple):
            out = out[0]
    return out.shape[1]


def _build_block(block_name: str, in_channels: int, args: list[Any]) -> nn.Module:
    cls = BLOCK_REGISTRY.get(block_name)
    if cls is None:
        raise ValueError(
            f"Block '{block_name}' không có trong BLOCK_REGISTRY.\n"
            f"Các block hợp lệ: {list(BLOCK_REGISTRY.keys())}"
        )
    try:
        return cls(in_channels, *args)
    except TypeError as e:
        raise TypeError(f"Lỗi khởi tạo {block_name}({[in_channels]+list(args)}): {e}") from e


# ──────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────

def build_from_yaml(yaml_path: str | Path, in_channels: int = 3) -> GLKANet:
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

    num_classes: int = cfg["nc"]

    # ── Parse head config ──────────────────────────────────────
    head_cfg = cfg.get("head", {})
    if not isinstance(head_cfg, dict):   # guard: YAML cũ dùng head là list
        head_cfg = {}

    dropout      = float(head_cfg.get("dropout",      0.2))
    mid_features = int(head_cfg.get("mid_features",   0))
    use_bn       = bool(head_cfg.get("use_bn",        False))

    # ── Build backbone ─────────────────────────────────────────
    backbone_layers: list[nn.Module] = []
    layer_channels:  list[int]       = []
    outputs_ch = [in_channels]

    for row in cfg["backbone"]:
        from_idx, repeats, block_name, args = row
        in_ch = outputs_ch[from_idx]

        for _ in range(repeats):
            block  = _build_block(block_name, in_ch, args)
            out_ch = _out_channels_of(block, in_ch, args)
            backbone_layers.append(block)
            layer_channels.append(out_ch)
            outputs_ch.append(out_ch)
            in_ch = out_ch

    # ── Build head ─────────────────────────────────────────────
    head = ClassifierHead(
        in_features  = layer_channels[-1],
        num_classes  = num_classes,
        dropout      = dropout,
        mid_features = mid_features,
        use_bn       = use_bn,
    )

    return GLKANet(
        backbone_layers=nn.ModuleList(backbone_layers),
        head=head,
        layer_channels=layer_channels,
    )