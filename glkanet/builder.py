"""glkanet/builder.py — đọc yaml config → build GLKANet động."""

from __future__ import annotations

import torch
import torch.nn as nn
import yaml
from pathlib import Path

try:
    from glkanet.blocks import BLOCK_REGISTRY, ConvBnRelu, EfficientBlock, ShuffleGLKABlock
except ImportError:
    try:
        from blocks import BLOCK_REGISTRY, ConvBnRelu, EfficientBlock, ShuffleGLKABlock
    except ImportError:
        # Dự phòng nếu registry được import trực tiếp từ file cục bộ
        BLOCK_REGISTRY = {}


# ──────────────────────────────────────────────────────────────
# Classifier head
# ──────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    GAP → Flatten → [BN1d] → [Linear(in→mid) + BN1d + Hardswish] → Dropout → Linear(→nc)
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

        if use_bn:
            layers.append(nn.BatchNorm1d(in_features))

        if mid_features > 0:
            layers += [
                nn.Linear(in_features, mid_features, bias=False),
                nn.BatchNorm1d(mid_features),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(mid_features, num_classes),
            ]
        else:
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
        """Quét toàn bộ mô hình và gọi switch_to_deploy() của mọi block con nếu có."""
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        return self

    def is_deployed(self) -> bool:
        """Kiểm tra xem mô hình đã gộp nhánh (reparam) chưa, tránh đệ quy vô hạn.

        Check TỔNG QUÁT theo attribute `reparam_conv` — KHÔNG hardcode tên class
        (khác bản cũ liệt kê "GLKA_Shuffle", "GLKA_SExCA"...). Nhờ vậy khi bạn
        thêm biến thể GLKA mới (GLKA_XYZ...) sau này, hàm này tự nhận diện đúng
        mà không cần sửa lại danh sách tên ở đây.
        """
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
    """Đọc số lượng đầu ra kênh (out_channels) một cách an toàn mà không bị gãy khi đổi vị trí tham số.

    LƯU Ý QUAN TRỌNG: nhánh nhanh (đầu tiên) chỉ hoạt động nếu block TỰ LƯU
    `self.out_channels` (hoặc `self.out_ch`) trong __init__. Nếu block mới bạn
    viết sau này quên lưu attribute này, hàm sẽ rơi xuống fallback forward-dummy
    (chậm hơn và có thể sai channel giả định), rồi tới fallback quét ngược module
    cuối cùng (dễ sai nếu switch_to_deploy() đổi cấu trúc module con).
    → Khi viết block mới: LUÔN thêm `self.out_channels = out_channels` trong __init__.
    """
    for attr in ("out_channels", "out_ch"):
        if hasattr(block, attr):
            val = getattr(block, attr)
            if isinstance(val, int):
                return val

    # Dự phòng (Fallback): forward thử nghiệm bằng ma trận 0 nếu không tìm thấy thuộc tính
    with torch.no_grad():
        in_ch = getattr(block, "in_channels", 16)
        try:
            dummy = torch.zeros(1, in_ch, 32, 32)
            out   = block(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.shape[1]
        except Exception:
            # Quét ngược các lớp layer cuối cùng để lấy cấu hình channel thực tế
            for m in reversed(list(block.modules())):
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    if hasattr(m, "out_channels"): return m.out_channels
                    if hasattr(m, "num_features"): return m.num_features
                    if hasattr(m, "out_features"): return m.out_features
    raise AttributeError(f"Không thể xác định out_channels của layer: {block.__class__.__name__}")


def _build_block(block_name: str, in_channels: int, args: list) -> nn.Module:
    """Khởi tạo block từ registry dựa trên mảng tham số (list) truyền thống từ YAML."""
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

    # ── Parse head config ──────────────────────────────────────
    head_cfg = cfg.get("head", {})
    if not isinstance(head_cfg, dict):
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
            out_ch = _out_channels_of(block)  # <--- Đọc động trực tiếp từ instance block vừa tạo

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