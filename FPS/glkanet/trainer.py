"""glkanet/trainer.py — Train loop, optimizer, scheduler, evaluate.

TỐI ƯU so với bản gốc (không đổi logic train/eval/checkpoint):
  1. AMP (torch.cuda.amp) — giảm tải GPU compute + giảm bộ nhớ, cho phép
     batch lớn hơn / nhanh hơn trên GPU hỗ trợ FP16/BF16.
  2. .to(device, non_blocking=True) — khai thác đúng pin_memory=True
     đã set trong dataloader.
  3. Bỏ gọi loss.item() ngay trong vòng lặp để hiển thị progress bar mỗi
     step (bản gốc gọi .item() 2 lần/step → 2 lần sync CPU-GPU/step).
     Thay vào đó tích lũy loss dạng tensor trên GPU, chỉ .item() 1 lần
     sau mỗi N step (giảm sync nhưng vẫn cập nhật progress bar mượt).
  4. torch.backends.cudnn.benchmark = True khi không cần reproducibility
     tuyệt đối (vẫn giữ set_seed, chỉ tắt deterministic gây chậm cudnn).
  5. set_to_none=True khi zero_grad — nhanh hơn việc set về 0 toàn bộ tensor.
"""

from __future__ import annotations

import random
import platform
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from glkanet.logger import (
    plot_curves, plot_confusion_matrix, plot_tsne,
    save_report, write_train_header,
)
from glkanet.exporter import export_all


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True + benchmark=False ép cudnn chọn algorithm chậm
    # nhưng ổn định bit-for-bit. Đổi sang benchmark=True để cudnn tự
    # profile và chọn kernel nhanh nhất cho shape input cố định của bạn
    # (ảnh 224x224 cố định → benchmark rất hiệu quả, an toàn cho training
    # thông thường, chỉ ảnh hưởng reproducibility tuyệt đối giữa các lần
    # chạy, không ảnh hưởng tính đúng đắn).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ──────────────────────────────────────────────────────────────
# Save dir
# ──────────────────────────────────────────────────────────────

def create_save_dir(runs_base: str | Path) -> Path:
    runs = Path(runs_base)
    runs.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = runs / f"exp{n}"
        if not d.exists():
            d.mkdir(parents=True)
            return d
        n += 1


# ──────────────────────────────────────────────────────────────
# Optimizer + Scheduler factory (giữ nguyên)
# ──────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, opt_cfg: dict) -> optim.Optimizer:
    t = opt_cfg["type"]
    if t == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 5e-4),
            nesterov=opt_cfg.get("nesterov", True),
        )
    if t == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
        )
    raise ValueError(f"Optimizer không hỗ trợ: {t}  (SGD | AdamW)")


def build_scheduler(
    optimizer: optim.Optimizer,
    sch_cfg:   dict,
    epochs:    int,
) -> optim.lr_scheduler.LRScheduler:
    t = sch_cfg["type"]
    if t == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=sch_cfg.get("eta_min", 1e-6),
        )
    if t == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 10),
            gamma=sch_cfg.get("gamma", 0.1),
        )
    raise ValueError(f"Scheduler không hỗ trợ: {t}  (CosineAnnealingLR | StepLR)")


# ──────────────────────────────────────────────────────────────
# Evaluate — AMP + non_blocking
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool = True,
    use_channels_last: bool = False,
) -> tuple[float, list, list, list]:
    """Trả về (avg_loss, preds, labels, features)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_feats = [], [], []

    pbar = tqdm(
        loader,
        desc="[Validating]",
        leave=False,
        bar_format='{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        if use_channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True).long()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, feats = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_feats.extend(feats.float().cpu().numpy())

    return total_loss / len(loader.dataset), all_preds, all_labels, all_feats


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────

class Trainer:
    """Quản lý toàn bộ train loop.

    Args:
        model:       GLKANet
        cfg:         dict parse từ ccmt.yaml (toàn bộ)
        save_dir:    Path thư mục exp
        class_names: list tên class từ dataloader
    """

    def __init__(
        self,
        model:       nn.Module,
        cfg:         dict,
        save_dir:    Path,
        class_names: List[str],
    ):
        self.model       = model
        self.cfg         = cfg
        self.save_dir    = Path(save_dir)
        self.class_names = class_names

        tr_cfg    = cfg["train"]
        hw_cfg    = cfg["hardware"]
        self.device   = torch.device(hw_cfg["device"])
        self.epochs   = tr_cfg["epochs"]
        self.log_cfg  = cfg.get("logging", {})
        self.exp_cfg  = cfg.get("export",  {})

        # AMP chỉ có ý nghĩa trên CUDA; tắt tự động nếu chạy CPU
        self.use_amp = hw_cfg.get("amp", True) and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.amp_dtype = torch.float16

        label_smoothing   = tr_cfg.get("label_smoothing", 0.0)
        self.criterion    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer    = build_optimizer(model, tr_cfg["optimizer"])
        self.scheduler    = build_scheduler(
            self.optimizer, tr_cfg["scheduler"], self.epochs)

        # channels_last là tối ưu QUAN TRỌNG cho depthwise/grouped conv
        # (groups=dim trong GLKA) — cuDNN có kernel riêng tối ưu hơn nhiều
        # cho conv depthwise khi tensor ở layout NHWC so với NCHW mặc định.
        self.use_channels_last = hw_cfg.get("channels_last", True) and self.device.type == "cuda"
        self.model.to(self.device)
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            print("  [trainer] channels_last bật — tối ưu depthwise/grouped conv (GLKA)")

        # torch.compile fuse nhiều kernel nhỏ (depthwise conv, BN, branches
        # trong GLKA) thành ít kernel lớn hơn → giảm mạnh overhead launch.
        # LƯU Ý: torch.compile cần Triton, mà Triton KHÔNG hỗ trợ tốt trên
        # Windows native (chỉ official trên Linux/WSL) → mặc định TẮT trên
        # Windows để tránh crash. Muốn dùng trên Windows: cài triton-windows
        # (pip install triton-windows) rồi set hardware.compile: true.
        is_windows = platform.system() == "Windows"
        self.use_compile = hw_cfg.get("compile", not is_windows) and self.device.type == "cuda"
        if self.use_compile:
            try:
                torch._dynamo.config.suppress_errors = True  # fallback về eager nếu compile lỗi giữa chừng
                self.model = torch.compile(self.model, mode="default")
                print("  [trainer] torch.compile bật (mode=default) — "
                      "epoch đầu sẽ chậm hơn do compile, các epoch sau nhanh hơn nhiều")
            except Exception as e:
                print(f"  [trainer] torch.compile thất bại, bỏ qua: {e}")
                self.use_compile = False
        else:
            reason = "Windows (Triton không hỗ trợ tốt)" if is_windows else "tắt qua config"
            print(f"  [trainer] torch.compile KHÔNG bật ({reason})")

        self.hist      = dict(train_loss=[], val_loss=[], val_f1=[], val_acc=[])
        self.best_f1   = 0.0
        self.best_loss = float("inf")

        if self.use_amp:
            print("  [trainer] AMP (mixed precision) bật — fp16 trên CUDA")

    # ── Run ───────────────────────────────────────────────────
    def run(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        model_yaml:   str | Path | None = None,
        cfg_path:     str | Path | None = None,
        data_yaml:    str | Path | None = None,
    ) -> dict:
        """Full train → val → test → export."""
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        write_train_header(
            self.save_dir, self.cfg,
            len(self.class_names), total_params, self.class_names,
        )

        tsne_interval = self.log_cfg.get("tsne_interval", 20)
        log_every = 10  # chỉ .item() loss mỗi N step để giảm sync GPU-CPU

        # ══════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════
        for epoch in range(self.epochs):
            lr_curr = self.optimizer.param_groups[0]['lr']

            self.model.train()
            # Tích lũy loss trên GPU dạng tensor, tránh .item() mỗi step
            train_loss_sum = torch.zeros(1, device=self.device)
            n_seen = 0

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                bar_format='{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                leave=True
            )

            for i, (imgs, labels) in pbar:
                imgs   = imgs.to(self.device, non_blocking=True)
                if self.use_channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True).long()

                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type,
                                     dtype=self.amp_dtype, enabled=self.use_amp):
                    logits, _ = self.model(imgs)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Tích lũy không sync (loss.detach() vẫn nằm trên GPU)
                train_loss_sum += loss.detach() * imgs.size(0)
                n_seen += imgs.size(0)

                # Chỉ đồng bộ (đọc giá trị về CPU) mỗi log_every step
                # hoặc ở step cuối cùng để cập nhật progress bar — giảm
                # mạnh số lần CPU phải đợi GPU so với bản gốc (mỗi step).
                if (i % log_every == 0) or (i == len(train_loader) - 1):
                    cur_loss = loss.item()
                    pbar.set_postfix(lr=f"{lr_curr:.2e}", loss=f"{cur_loss:.4f}")

            train_loss = (train_loss_sum / n_seen).item()

            # ── Val ──
            val_loss, val_preds, val_labels, val_feats = evaluate(
                self.model, val_loader, self.criterion, self.device,
                use_amp=self.use_amp, use_channels_last=self.use_channels_last,
            )
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)

            self.hist["train_loss"].append(train_loss)
            self.hist["val_loss"].append(val_loss)
            self.hist["val_f1"].append(val_f1)
            self.hist["val_acc"].append(val_acc * 100)

            self.scheduler.step()

            print(f"       ↳ Summary: loss={train_loss:.4f} | val_loss={val_loss:.4f} | acc={val_acc*100:.2f}% | f1={val_f1:.4f}")

            save_report(
                val_preds, val_labels, self.class_names, self.save_dir,
                tag="val", epoch=epoch,
                train_loss=train_loss, val_loss=val_loss, val_f1=val_f1,
            )

            # ── Checkpoint: best F1 ──
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self._save_ckpt("best_f1.pt", epoch, val_f1=val_f1, val_acc=val_acc)
                print(f"       [↑ F1] New best F1: {self.best_f1:.4f}")
                plot_confusion_matrix(
                    val_preds, val_labels, self.class_names,
                    self.save_dir, "val_best_f1",
                )
                save_report(
                    val_preds, val_labels, self.class_names,
                    self.save_dir, tag="val_best_f1",
                )
                if tsne_interval > 0 and (epoch + 1) % tsne_interval == 0:
                    plot_tsne(
                        val_feats, val_labels, self.class_names,
                        self.save_dir, f"val_ep{epoch+1}",
                    )

            # ── Checkpoint: best loss ──
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_ckpt("best_loss.pt", epoch, val_loss=val_loss)
                print(f"       [↓ Loss] New best Loss: {self.best_loss:.4f}")

            plot_curves(
                self.hist["train_loss"], self.hist["val_loss"],
                self.hist["val_f1"],    self.hist["val_acc"],
                self.save_dir,
            )

        # ══════════════════════════════════════════════════════
        # TEST SET
        # ══════════════════════════════════════════════════════
        if self.log_cfg.get("eval_test", True) and test_loader is not None:
            print("\n[*] Test set evaluation (best_f1.pt)...")
            self._load_ckpt("best_f1.pt")
            test_loss, test_preds, test_labels, test_feats = evaluate(
                self.model, test_loader, self.criterion, self.device,
                use_amp=self.use_amp, use_channels_last=self.use_channels_last,
            )
            test_acc, test_f1 = save_report(
                test_preds, test_labels, self.class_names,
                self.save_dir, tag="test",
            )
            plot_confusion_matrix(
                test_preds, test_labels, self.class_names, self.save_dir, "test")
            plot_tsne(
                test_feats, test_labels, self.class_names, self.save_dir, "test")
            print(f"  acc={test_acc*100:.2f}%  f1={test_f1:.4f}")

        # ══════════════════════════════════════════════════════
        # EXPORT
        # ══════════════════════════════════════════════════════
        if self.exp_cfg.get("enabled", True):
            print("\n[*] Exporting weights...")
            self._load_ckpt("best_f1.pt")
            self.model.eval()
            export_all(
                model      = self.model,
                save_dir   = self.save_dir,
                input_size = self.cfg.get("img_size", 224),
                yaml_path  = cfg_path,
                opset      = self.exp_cfg.get("opset", 18),
                verbose    = True,
            )

        print(f"\n{'='*60}")
        print(f"[✓] Done  best_f1={self.best_f1:.4f}  best_loss={self.best_loss:.4f}")
        print(f"[✓] {self.save_dir}")
        print(f"{'='*60}\n")

        return {
            "best_f1":   self.best_f1,
            "best_loss": self.best_loss,
            "save_dir":  self.save_dir,
        }

    # ── Helpers ───────────────────────────────────────────────
    def _save_ckpt(self, name: str, epoch: int, **metrics) -> None:
        torch.save(
            {"state_dict": self.model.state_dict(), "epoch": epoch, **metrics},
            self.save_dir / name,
        )

    def _load_ckpt(self, name: str) -> None:
        ckpt = torch.load(
            self.save_dir / name,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt["state_dict"])