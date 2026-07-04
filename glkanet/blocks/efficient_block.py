import torch
import torch.nn as nn

try:
    from glkanet.blocks.conv_blocks  import conv_bn_relu
    from glkanet.blocks.glka_shuffle import GLKA_Shuffle
    from glkanet.blocks.glka_SExCA   import GLKA_SExCA
    from glkanet.blocks.glka_CA_SE   import GLKA_CA_SE
except ImportError:
    from .conv_blocks  import conv_bn_relu
    from .glka_shuffle import GLKA_Shuffle
    from .glka_SExCA   import GLKA_SExCA
    from .glka_CA_SE   import GLKA_CA_SE

GLKA_VARIANTS: dict[str, type] = {
    "shuffle":  GLKA_Shuffle,  # conv0 -> split -> SE / dilated -> concat -> shuffle -> fuse 1x1 groups=2
    "sexca":    GLKA_SExCA,    # SE(anchor) * branch_sum
    "casexse":  GLKA_CA_SE,    # SE(branch_sum)
}
GLKA_CLASSES = tuple(GLKA_VARIANTS.values())
DEFAULT_GLKA_VARIANT = "shuffle"   # <-- đổi ở đây nếu muốn default khác


class EfficientBlock(nn.Module):

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        stride:          int,
        expansion_ratio: int  = 2,
        use_glka:        bool = True,
        glka_variant:    str  = DEFAULT_GLKA_VARIANT,
        glka_K:          int  = 13,
        se_reduction:    int  = 0,
        no_residual:     bool = False,
    ):
        super().__init__()
        self.in_channels  = in_channels   # để builder._out_channels_of() đọc thẳng, khỏi đoán
        self.out_channels = out_channels  # ---
        self.stride       = stride
        self.glka_variant = glka_variant
        self.use_glka     = use_glka
        hidden_dim  = in_channels * expansion_ratio

        self.use_residual = (stride == 1) and not no_residual

        # ── Expand ────────────────────────────────────────────────────
        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)

        # ── Depthwise / GLKA ──────────────────────────────────────────
        if use_glka:
            if glka_variant not in GLKA_VARIANTS:
                raise ValueError(f"glka_variant='{glka_variant}' không hợp lệ.")
            glka_cls  = GLKA_VARIANTS[glka_variant]
            self.dw   = nn.Identity()

            self.glka = glka_cls(
                dim          = hidden_dim,
                out_channels = out_channels,
                K            = glka_K,
                stride       = stride,
                se_reduction = se_reduction,
            )
        else:
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=stride, padding=1, groups=hidden_dim)
            self.glka = nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # ── Shortcut ─────────────────────────────────────────────────
        if self.use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)

        if self.use_residual:
            identity = self.shortcut(x)
            return identity + out
        return out

    def switch_to_deploy(self) -> None:
        """Gọi reparam + fold BN/shuffle cho GLKA con (nếu có switch_to_deploy)."""
        if hasattr(self.glka, "switch_to_deploy"):
            self.glka.switch_to_deploy()

    def __repr__(self) -> str:
        glka_info = repr(self.glka) if isinstance(self.glka, GLKA_CLASSES) else "off"
        return (
            f"EfficientBlock("
            f"residual={self.use_residual}, "
            f"stride={self.stride}, "
            f"variant={self.glka_variant}, "
            f"glka={glka_info})"
        )