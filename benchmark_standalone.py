"""
benchmark_standalone.py — 1 file DUY NHẤT: định nghĩa model (từ các block đã
gửi) + build từ config backbone + đo tốc độ bằng thư viện chính chủ
(torch.utils.benchmark.Timer cho PyTorch/TorchScript, pattern chuẩn ORT cho
ONNX Runtime). Không cần import package glkanet — chạy độc lập.

Cách dùng nhanh:
    pip install torch onnxruntime
    python benchmark_standalone.py

Nếu có checkpoint thật, set WEIGHTS_PATH bên dưới (không bắt buộc — không có
thì model chạy random-init, không ảnh hưởng đến việc đo LATENCY).
"""

from __future__ import annotations
import sys
import time
import statistics

import torch
import torch.nn as nn
import torch.utils.benchmark as torch_benchmark

# ══════════════════════════════════════════════════════════════════════
# PHẦN 1 — CÁC BLOCK MODEL (gộp nguyên từ code bạn gửi, bỏ import package)
# ══════════════════════════════════════════════════════════════════════

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0,
                  groups=1, activation=True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                   padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=-1, groups=1, activation=True):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        super().__init__(*conv_bn_relu(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        groups=groups, activation=activation))


class SEBlock(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


GLKA_PRESETS = {
    13: [(3, 1), (3, 3), (5, 2), (5, 3), (13, 1)],
    7:  [(7, 1), (3, 2), (3, 3)],
    5:  [(5, 1), (3, 2)],
    3:  [(3, 1), (1, 1)],
}


class GLKA_CBAM(nn.Module):
    """Dùng bên trong ShuffleGLKABlock (nhánh proc)."""
    def __init__(self, dim, K=13, stride=1, conv0_k=5, branches_config=None, se_reduction=4):
        super().__init__()
        self.dim, self.K, self.stride, self.conv0_k = dim, K, stride, conv0_k
        self.branches_config = [tuple(b) for b in branches_config] if branches_config else GLKA_PRESETS[K]

        pad0 = conv0_k // 2
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, conv0_k, stride=stride, padding=pad0, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU6(inplace=True),
        )
        hidden = max(1, dim // se_reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, dim, 1, bias=True), nn.Sigmoid(),
        )
        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, k_size, padding=pad, groups=dim, dilation=dil, bias=False),
                nn.BatchNorm2d(dim),
            ))
        self.spatial_gate_act = nn.Sigmoid()
        self.reparam_conv = None
        self._deployed = False

    def forward(self, x):
        anchor = self.conv0(x)
        se_gate = self.se(anchor)
        anchor_se = anchor * se_gate
        if self.reparam_conv is not None:
            spatial_gate = self.reparam_conv(anchor_se)
        else:
            branch_outs = [b(anchor_se) for b in self.branches]
            spatial_gate = torch.stack(branch_outs, dim=0).sum(dim=0)
        spatial_gate = self.spatial_gate_act(spatial_gate)
        return anchor_se * spatial_gate

    def switch_to_deploy(self):
        if self._deployed:
            return
        if hasattr(self, "branches"):
            W_equiv, B_equiv = 0, 0
            for branch, (k_size, dil) in zip(self.branches, self.branches_config):
                w_fused, b_fused = self._fuse_bn(branch)
                W_equiv += self._to_target_k(w_fused, k_size, dil)
                B_equiv += b_fused
            self.reparam_conv = nn.Conv2d(self.dim, self.dim, self.K, padding=self.K // 2,
                                           groups=self.dim, bias=True)
            self.reparam_conv.weight.data = W_equiv
            self.reparam_conv.bias.data = B_equiv
            del self.branches
        w0, b0 = self._fuse_bn(self.conv0[:2])
        new_conv0 = nn.Conv2d(self.dim, self.dim, self.conv0_k, stride=self.stride,
                               padding=self.conv0_k // 2, groups=self.dim, bias=True)
        new_conv0.weight.data = w0
        new_conv0.bias.data = b0
        self.conv0 = nn.Sequential(new_conv0, nn.ReLU6(inplace=True))
        self._deployed = True

    def _fuse_bn(self, block):
        conv, bn = block[0], block[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        return conv.weight * t, bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)

    def _to_target_k(self, kernel, orig_k, d):
        c, m = kernel.shape[:2]
        kd = (orig_k - 1) * d + 1
        out = torch.zeros((c, m, self.K, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out


def channel_shuffle_v1(x, groups=2):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


def channel_shuffle_v2(x, groups=2):
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    x = x.reshape(-1, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(-1, c, h, w)


class GLKA_Shuffle(nn.Module):
    """Dùng bên trong EfficientBlock (variant='shuffle')."""
    def __init__(self, dim, out_channels, K=13, stride=1, branches_config=None, se_reduction=8):
        super().__init__()
        self.dim, self.out_channels, self.K, self.stride = dim, out_channels, K, stride
        self.mid_dim = dim // 2
        self.branches_config = [tuple(b) for b in branches_config] if branches_config else GLKA_PRESETS[K]

        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, stride=stride, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU6(inplace=True),
        )
        self.se = SEBlock(self.mid_dim, reduction=se_reduction) if se_reduction > 0 else None
        self.branches = nn.ModuleList()
        for k_size, dil in self.branches_config:
            pad = ((k_size - 1) * dil) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(self.mid_dim, self.mid_dim, k_size, padding=pad, groups=self.mid_dim,
                          dilation=dil, bias=False),
                nn.BatchNorm2d(self.mid_dim),
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, out_channels, 1, groups=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.reparam_conv = None
        self._deployed = False

    def forward(self, x):
        anchor = self.conv0(x)
        x_left, x_right = torch.chunk(anchor, chunks=2, dim=1)
        ca = self.se(x_left) if self.se is not None else x_left
        if self.reparam_conv is not None:
            sa = self.reparam_conv(x_right)
        else:
            sa = sum(b(x_right) for b in self.branches)
        out = torch.cat([ca, sa], dim=1)
        out = channel_shuffle_v2(out, groups=2)
        return self.fuse(out)

    def switch_to_deploy(self):
        if self._deployed:
            return
        if hasattr(self, "branches"):
            W_equiv, B_equiv = 0, 0
            for branch, (k_size, dil) in zip(self.branches, self.branches_config):
                w_fused, b_fused = self._fuse_bn(branch)
                W_equiv += self._to_target_k(w_fused, k_size, dil)
                B_equiv += b_fused
            self.reparam_conv = nn.Conv2d(self.mid_dim, self.mid_dim, self.K, padding=self.K // 2,
                                           groups=self.mid_dim, bias=True)
            self.reparam_conv.weight.data = W_equiv
            self.reparam_conv.bias.data = B_equiv
            del self.branches
        w0, b0 = self._fuse_bn(self.conv0[:2])
        new_conv0 = nn.Conv2d(self.dim, self.dim, 5, stride=self.stride, padding=2,
                               groups=self.dim, bias=True)
        new_conv0.weight.data = w0
        new_conv0.bias.data = b0
        self.conv0 = nn.Sequential(new_conv0, nn.ReLU6(inplace=True))
        # channel_shuffle KHÔNG fold được vào fuse (groups=2 conv không biểu diễn được
        # phép trộn xuyên-group) — vẫn chạy runtime kể cả ở deploy mode.
        wf, bf = self._fuse_bn(self.fuse)
        new_fuse = nn.Conv2d(self.dim, self.out_channels, 1, groups=2, bias=True)
        new_fuse.weight.data = wf
        new_fuse.bias.data = bf
        self.fuse = new_fuse
        self._deployed = True

    def _fuse_bn(self, block):
        conv, bn = block[0], block[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        return conv.weight * t, bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)

    def _to_target_k(self, kernel, orig_k, d):
        c, m = kernel.shape[:2]
        kd = (orig_k - 1) * d + 1
        out = torch.zeros((c, m, self.K, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out


GLKA_VARIANTS = {"shuffle": GLKA_Shuffle}  # chỉ giữ variant dùng trong yaml config


class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=2,
                 use_glka=True, glka_variant="shuffle", glka_K=13, se_reduction=0,
                 no_residual=False):
        super().__init__()
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        self.glka_variant, self.use_glka = glka_variant, use_glka
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = (stride == 1) and not no_residual
        self._deployed = False

        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)

        if use_glka:
            if glka_variant not in GLKA_VARIANTS:
                raise ValueError(f"glka_variant='{glka_variant}' không hỗ trợ trong bản standalone này.")
            glka_cls = GLKA_VARIANTS[glka_variant]
            self.dw = nn.Identity()
            self.glka = glka_cls(dim=hidden_dim, out_channels=out_channels, K=glka_K,
                                  stride=stride, se_reduction=se_reduction)
        else:
            self.dw = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                    padding=1, groups=hidden_dim)
            self.glka = nn.Sequential(nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels))

        if self.use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                           nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        if self.use_residual:
            return self.shortcut(x) + out
        return out

    def switch_to_deploy(self):
        if self._deployed:
            return
        if hasattr(self.glka, "switch_to_deploy"):
            self.glka.switch_to_deploy()
        if isinstance(self.expand, nn.Sequential) and self._seq_is_conv_bn(self.expand):
            self.expand = self._fold_conv_bn_seq(self.expand)
        if isinstance(self.dw, nn.Sequential) and self._seq_is_conv_bn(self.dw):
            self.dw = self._fold_conv_bn_seq(self.dw)
        if isinstance(self.shortcut, nn.Sequential) and self._seq_is_conv_bn(self.shortcut):
            self.shortcut = self._fold_conv_bn_seq(self.shortcut)
        self._deployed = True

    @staticmethod
    def _seq_is_conv_bn(seq):
        return len(seq) >= 2 and isinstance(seq[0], nn.Conv2d) and seq[0].bias is None and isinstance(seq[1], nn.BatchNorm2d)

    @staticmethod
    def _fold_conv_bn_seq(seq):
        conv, bn = seq[0], seq[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)
        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                              stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True)
        new_conv.weight.data = w_fused
        new_conv.bias.data = b_fused
        return nn.Sequential(new_conv, *list(seq[2:]))


class ShuffleGLKABlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, split_ratio=0.5,
                 use_glka=True, glka_K=13, se_reduction=0):
        super().__init__()
        self.stride, self.in_channels, self.out_channels = stride, in_channels, out_channels
        self.use_glka = use_glka
        self._deployed = False

        if stride == 1:
            assert in_channels == out_channels
            self.proc_dim = max(1, int(round(in_channels * split_ratio)))
            self.id_dim = in_channels - self.proc_dim
            proc_in, proc_out = self.proc_dim, self.proc_dim
            id_in = self.id_dim
        else:
            assert out_channels % 2 == 0
            self.proc_dim = out_channels // 2
            self.id_dim = out_channels - self.proc_dim
            proc_in, proc_out = in_channels, self.proc_dim
            id_in = in_channels

        if use_glka:
            self._proc_needs_proj = (proc_in != proc_out)
            if self._proc_needs_proj:
                self.proc_proj = nn.Sequential(nn.Conv2d(proc_in, proc_out, 1, bias=False),
                                                nn.BatchNorm2d(proc_out), nn.ReLU6(inplace=True))
            else:
                self.proc_proj = nn.Identity()
            self.proc = GLKA_CBAM(dim=proc_out, K=glka_K, stride=stride, se_reduction=se_reduction)
        else:
            self._proc_needs_proj = False
            self.proc_proj = nn.Identity()
            self.proc = conv_bn_relu(proc_in, proc_out, kernel_size=3, stride=stride, padding=1,
                                      groups=proc_in if proc_in == proc_out else 1)

        if stride == 1:
            self.id_branch = nn.Identity()
        else:
            self.id_branch = nn.Sequential(
                nn.Conv2d(id_in, id_in, 3, stride=stride, padding=1, groups=id_in, bias=False),
                nn.BatchNorm2d(id_in), nn.ReLU6(inplace=True),
                nn.Conv2d(id_in, self.id_dim, 1, bias=False), nn.BatchNorm2d(self.id_dim),
            )

    def forward(self, x):
        if self.stride == 1:
            x_id, x_proc = x[:, :self.id_dim], x[:, self.id_dim:]
        else:
            x_id, x_proc = x, x
        identity = self.id_branch(x_id)
        proc = self.proc(self.proc_proj(x_proc))
        out = torch.cat([identity, proc], dim=1)
        return channel_shuffle_v1(out, groups=2)

    def switch_to_deploy(self):
        if self._deployed:
            return
        if self.use_glka and hasattr(self.proc, "switch_to_deploy"):
            self.proc.switch_to_deploy()
        if self.stride != 1 and isinstance(self.id_branch, nn.Sequential):
            self.id_branch = self._fold_sequential_bn(self.id_branch, pairs=[(0, 1), (3, 4)],
                                                       relu_positions={2: nn.ReLU6(inplace=True)})
        if self._proc_needs_proj and isinstance(self.proc_proj, nn.Sequential):
            self.proc_proj = self._fold_sequential_bn(self.proc_proj, pairs=[(0, 1)],
                                                        relu_positions={2: nn.ReLU6(inplace=True)})
        self._deployed = True

    @staticmethod
    def _fold_sequential_bn(seq, pairs, relu_positions):
        new_layers = {}
        max_idx = max(max(p) for p in pairs)
        if relu_positions:
            max_idx = max(max_idx, max(relu_positions.keys()))
        for conv_idx, bn_idx in pairs:
            conv, bn = seq[conv_idx], seq[bn_idx]
            std = (bn.running_var + bn.eps).sqrt()
            t = (bn.weight / std).reshape(-1, 1, 1, 1)
            b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
            w_fused = conv.weight * t
            b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
            new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                  stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True)
            new_conv.weight.data = w_fused
            new_conv.bias.data = b_fused
            new_layers[conv_idx] = new_conv
        for pos, act in relu_positions.items():
            new_layers[pos] = act
        ordered = [new_layers[i] for i in range(max_idx + 1) if i in new_layers]
        return nn.Sequential(*ordered)


# ══════════════════════════════════════════════════════════════════════
# PHẦN 2 — BUILDER: dựng model trực tiếp từ backbone config (đúng file yaml)
# ══════════════════════════════════════════════════════════════════════

# Backbone config y hệt file Hybird.yaml đã gửi. Mỗi dòng: [from, repeat, module, args]
# args luôn bắt đầu bằng out_channels; in_channels tự lấy từ output kênh trước đó.
BACKBONE_CONFIG = [
    [-1, 1, "ConvBnRelu",       [32,  3, 2, 1, 1, False]],
    [-1, 1, "ShuffleGLKABlock", [64,  2, 0.25, True, 3, 8]],
    [-1, 1, "ShuffleGLKABlock", [64,  1, 0.25, True, 3, 8]],
    [-1, 1, "EfficientBlock",   [64,  1, 2, True, "shuffle", 5, 16, True]],
    [-1, 1, "ShuffleGLKABlock", [128, 2, 0.75, True, 3, 8]],
    [-1, 1, "EfficientBlock",   [128, 1, 2, True, "shuffle", 5, 16, True]],
    [-1, 1, "ShuffleGLKABlock", [256, 2, 0.75, True, 5, 8]],
    [-1, 1, "ShuffleGLKABlock", [512, 2, 0.75, True, 7, 8]],
]
NUM_CLASSES = 22


class GLKANetStandalone(nn.Module):
    """Dựng backbone trực tiếp từ BACKBONE_CONFIG + head classification đơn giản."""

    MODULE_MAP = {
        "ConvBnRelu": ConvBnRelu,
        "ShuffleGLKABlock": ShuffleGLKABlock,
        "EfficientBlock": EfficientBlock,
    }

    def __init__(self, backbone_config=BACKBONE_CONFIG, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        layers = []
        in_channels = 3
        for _from, repeat, module_name, args in backbone_config:
            cls = self.MODULE_MAP[module_name]
            out_channels = args[0]
            for _ in range(repeat):
                layers.append(cls(in_channels, *args))
                in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def switch_to_deploy(self):
        for m in self.backbone:
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()

    def is_deployed(self):
        return all(getattr(m, "_deployed", True) for m in self.backbone)


# ══════════════════════════════════════════════════════════════════════
# PHẦN 3 — BENCHMARK CHÍNH CHỦ (torch.utils.benchmark.Timer + pattern ORT)
# ══════════════════════════════════════════════════════════════════════

WEIGHTS_PATH = None   # set path .pt nếu có checkpoint thật, để None nếu chỉ test tốc độ
DEVICE       = "auto" # "auto" | "cuda" | "cpu"
BATCH        = 1      # batch=1 = chuẩn latency real-time cho paper edge-AI
IMG_SIZE     = 224
SEED         = 42

MIN_RUN_TIME = 3.0
NUM_THREADS  = None

RUN_ONNX     = True
ONNX_OPSET   = 17
ONNX_WARMUP  = 50
ONNX_STEPS   = 500
ONNX_REPEATS = 5


def resolve_device(pref):
    pref = (pref or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_model(device):
    model = GLKANetStandalone().to(device)
    if WEIGHTS_PATH:
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.switch_to_deploy()
    return model


def bench_with_official_timer(forward_callable, globals_dict, device, label):
    timer = torch_benchmark.Timer(
        stmt="fn()", globals={"fn": forward_callable, **globals_dict},
        num_threads=NUM_THREADS or torch.get_num_threads(), label=label,
        sub_label=f"batch={BATCH}", description=str(device),
    )
    return timer.blocked_autorange(min_run_time=MIN_RUN_TIME)


def print_measurement(name, measurement, batch):
    median_s, iqr_s = measurement.median, measurement.iqr
    fps = batch / median_s
    print(f"{name:<32} median={median_s*1000:9.4f} ms   IQR={iqr_s*1000:7.4f} ms   FPS(median)={fps:9.1f}")


def bench_onnxruntime(onnx_path, x_np, device, batch):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] Chưa cài onnxruntime — bỏ qua. pip install onnxruntime")
        return None

    sess_options = ort.SessionOptions()
    if NUM_THREADS:
        sess_options.intra_op_num_threads = NUM_THREADS
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    results = {}

    def run_provider(providers, key, label):
        try:
            sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        except Exception as ex:
            print(f"[warn] {label}: {ex}")
            return
        in_name = sess.get_inputs()[0].name
        for _ in range(ONNX_WARMUP):
            sess.run(None, {in_name: x_np})
        all_lat = []
        for _ in range(ONNX_REPEATS):
            for _ in range(ONNX_STEPS):
                t0 = time.perf_counter()
                sess.run(None, {in_name: x_np})
                all_lat.append((time.perf_counter() - t0) * 1000.0)
        arr = sorted(all_lat)
        n = len(arr)
        median = statistics.median(arr)
        results[key] = {
            "median_ms": median, "std_ms": statistics.pstdev(arr),
            "p90_ms": arr[min(int(0.9 * (n - 1)), n - 1)],
            "fps_median": batch / (median / 1000.0),
        }
        s = results[key]
        print(f"{label:<32} median={s['median_ms']:9.4f} ms   std={s['std_ms']:7.4f} ms   "
              f"p90={s['p90_ms']:8.4f}   FPS(median)={s['fps_median']:9.1f}")

    if "CPUExecutionProvider" in available:
        run_provider(["CPUExecutionProvider"], "onnx_cpu", "ONNX Runtime (CPU)")
    if device.type == "cuda" and "CUDAExecutionProvider" in available:
        run_provider(["CUDAExecutionProvider", "CPUExecutionProvider"], "onnx_cuda", "ONNX Runtime (CUDA)")
    return results


def main():
    torch.manual_seed(SEED)
    device = resolve_device(DEVICE)
    print(f"[info] PyTorch {torch.__version__} | Device: {device} | Batch: {BATCH}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[info] GPU: {torch.cuda.get_device_name(device)}")

    model = build_model(device)
    model.eval()
    print(f"[info] is_deployed = {model.is_deployed()}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Tổng params: {n_params/1e6:.3f} M")

    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=device)

    print(f"\n{'='*100}\nBATCH = {BATCH}\n{'='*100}")

    with torch.inference_mode():
        m_eager = bench_with_official_timer(lambda: model(x), {"model": model, "x": x}, device, "PyTorch eager")
    print_measurement("PyTorch eager (.pt)", m_eager, BATCH)
    baseline_median = m_eager.median

    ts_median = None
    with torch.inference_mode():
        try:
            ts_model = torch.jit.freeze(torch.jit.trace(model, x))
            m_ts = bench_with_official_timer(lambda: ts_model(x), {"ts_model": ts_model, "x": x}, device, "TorchScript")
            print_measurement("TorchScript (traced+frozen)", m_ts, BATCH)
            ts_median = m_ts.median
        except Exception as ex:
            print(f"[warn] Trace TorchScript thất bại: {ex}")

    onnx_results = None
    if RUN_ONNX:
        onnx_path = "GLKANet_standalone.onnx"
        try:
            export_x = x if device.type == "cuda" else torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
            model_for_export = model if device.type == "cuda" else model.to("cpu")
            torch.onnx.export(model_for_export, export_x, onnx_path,
                               input_names=["input"], output_names=["output"],
                               opset_version=ONNX_OPSET,
                               dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            model.to(device)
            x_np = x.detach().to("cpu").float().numpy()
            onnx_results = bench_onnxruntime(onnx_path, x_np, device, BATCH)
        except Exception as ex:
            print(f"[warn] Export/benchmark ONNX thất bại: {ex}")

    print(f"\nSo với PyTorch eager (baseline, median={baseline_median*1000:.4f} ms, device={device}):")
    if ts_median is not None:
        diff = (1 - ts_median / baseline_median) * 100
        print(f"  - TorchScript (cùng device {device}): {'nhanh hơn' if diff > 0 else 'chậm hơn'} {abs(diff):.1f}%")
    if onnx_results:
        for key, s in onnx_results.items():
            same_device = "cpu" in key if device.type == "cpu" else "cuda" in key
            tag = "cùng device" if same_device else "⚠ KHÁC device — không so trực tiếp"
            diff = (1 - (s["median_ms"] / 1000.0) / baseline_median) * 100
            print(f"  - {key:<12} ({tag}): {'nhanh hơn' if diff > 0 else 'chậm hơn'} {abs(diff):.1f}%")


if __name__ == "__main__":
    main()
