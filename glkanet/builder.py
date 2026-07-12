"""glkanet/builder.py — đọc yaml config → build GLKANet động."""

from __future__ import annotations

import torch
import torch.nn as nn
import yaml
from pathlib import Path

try:
    from glkanet.blocks import BLOCK_REGISTRY, ConvBnRelu, Dual_Attention_Block, ShuffleGLKABlock
except ImportError:
    try:
        from blocks import BLOCK_REGISTRY, ConvBnRelu, Dual_Attention_Block, ShuffleGLKABlock
    except ImportError:
        # Dự phòng nếu registry được import trực tiếp từ file cục bộ
        BLOCK_REGISTRY = {}


# ──────────────────────────────────────────────────────────────
# Classifier head
# ──────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Head cố định, đơn giản:
        GAP → Flatten → BatchNorm1d → Dropout → Linear(in_features → num_classes)

    Không còn nhánh mid_features/bottleneck, không còn cờ use_bn — BN1d luôn
    bật để chuẩn hoá feature vector trước Linear. switch_to_deploy() fold
    BN1d vào Linear thành 1 lớp Linear duy nhất (bias=True) cho deploy.
    """
    def __init__(
        self,
        in_features:  int,
        num_classes:  int,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.bn      = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(in_features, num_classes)

        self._deployed = False

    def forward(self, x: torch.Tensor):
        x        = self.pool(x)
        features = self.flatten(x)
        out      = self.bn(features)
        out      = self.dropout(out)
        logits   = self.fc(out)
        return logits, features

    def switch_to_deploy(self) -> None:
        """Fold BatchNorm1d vào Linear: BN(x) -> Linear thành 1 Linear duy nhất
        (Dropout ở eval() vốn đã là no-op nên không cần xử lý riêng)."""
        if self._deployed:
            return

        std   = (self.bn.running_var + self.bn.eps).sqrt()
        gamma = self.bn.weight / std                      # (in_features,)
        beta  = self.bn.bias - self.bn.running_mean * gamma  # (in_features,)

        new_fc = nn.Linear(self.fc.in_features, self.fc.out_features)
        # y = W(gamma*x + beta) + b = (W*gamma) x + (W@beta + b)
        new_fc.weight.data = self.fc.weight * gamma.unsqueeze(0)
        new_fc.bias.data   = self.fc.weight @ beta + self.fc.bias

        self.bn = nn.Identity()
        self.fc = new_fc
        self._deployed = True


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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.backbone_layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        return self.head(feat)

    def switch_to_deploy(self) -> "GLKANet":
        """Quét toàn bộ mô hình và gọi switch_to_deploy() của mọi block con nếu có."""
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        return self

    def is_deployed(self) -> bool:
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "reparam_conv") and m.reparam_conv is None:
                return False
        return True

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

def _out_channels_of(block: nn.Module) -> int:
    for attr in ("out_channels", "out_ch"):
        if hasattr(block, attr):
            val = getattr(block, attr)
            if isinstance(val, int):
                return val

    with torch.no_grad():
        in_ch = getattr(block, "in_channels", 16)
        try:
            dummy = torch.zeros(1, in_ch, 32, 32)
            out   = block(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.shape[1]
        except Exception:
            for m in reversed(list(block.modules())):
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    if hasattr(m, "out_channels"): return m.out_channels
                    if hasattr(m, "num_features"): return m.num_features
                    if hasattr(m, "out_features"): return m.out_features
    raise AttributeError(f"Không thể xác định out_channels của layer: {block.__class__.__name__}")


def _build_block(block_name: str, in_channels: int, args: list) -> nn.Module:
    cls = BLOCK_REGISTRY.get(block_name)
    if cls is None:
        raise ValueError(
            f"Block '{block_name}' không tồn tại trong BLOCK_REGISTRY.\n"
            f"Các lựa chọn hợp lệ: {list(BLOCK_REGISTRY.keys())}"
        )
    try:
        return cls(in_channels, *args)
    except TypeError as e:
        raise TypeError(f"Lỗi truyền tham số khi khởi tạo {block_name} (in_channels={in_channels}, args={args}): {e}") from e


# ──────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────

def build_from_yaml(yaml_path: str | Path, in_channels: int = 3) -> GLKANet:
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

    num_classes: int = cfg["nc"]

    # ── Parse head config (chỉ còn dropout) ────────────────────
    head_cfg = cfg.get("head", {})
    if not isinstance(head_cfg, dict):
        head_cfg = {}

    dropout = float(head_cfg.get("dropout", 0.2))

    # ── Build backbone ─────────────────────────────────────────
    backbone_layers: list[nn.Module] = []
    layer_channels:  list[int]       = []
    outputs_ch = [in_channels]

    for row in cfg["backbone"]:
        from_idx, repeats, block_name, args = row
        in_ch = outputs_ch[from_idx]

        for _ in range(repeats):
            block  = _build_block(block_name, in_ch, args)
            out_ch = _out_channels_of(block)

            backbone_layers.append(block)
            layer_channels.append(out_ch)
            outputs_ch.append(out_ch)
            in_ch = out_ch

    # ── Build head ─────────────────────────────────────────────
    head = ClassifierHead(
        in_features = layer_channels[-1],
        num_classes = num_classes,
        dropout     = dropout,
    )

    return GLKANet(
        backbone_layers=nn.ModuleList(backbone_layers),
        head=head,
        layer_channels=layer_channels,
    )