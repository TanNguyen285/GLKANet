"""Các hàm format số để hiển thị."""


def fmt_flops(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MFLOPs"
    return f"{n / 1e3:.2f} KFLOPs"


def fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return str(n)
