import torch.nn as nn

try:
    from glkanet.blocks.glka_shuffle import GLKA_Shuffle
except ImportError:
    from .glka_shuffle import GLKA_Shuffle


class Dual_Attention_Block(nn.Module):
    """Wrapper mỏng quanh GLKA_Shuffle — không residual, không expand riêng.
    GLKA_Shuffle tự quyết định tách/không tách kênh dựa theo stride, và tự
    raise lỗi nếu out_channels truyền vào không khớp công thức bắt buộc
    (2*dim nếu stride=2, dim nếu stride=1)."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int,
        glka_K:       int = 13,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.stride        = stride
        self._deployed      = False

        self.glka = GLKA_Shuffle(
            dim          = in_channels,
            out_channels = out_channels,
            K            = glka_K,
            stride       = stride,
        )
        self.out_channels = self.glka.out_channels  # đọc lại giá trị thực tế đã qua validate

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