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
  6. Resume training — checkpoint lưu cả optimizer/scheduler/scaler state,
     không chỉ model weights, để tiếp tục train đúng từ chỗ dừng (LR,
     momentum không bị reset về giá trị khởi tạo).
  7. Copy yaml kiến trúc model ra weights/ NGAY LÚC BẮT ĐẦU TRAIN (không
     đợi tới lúc export_all() ở cuối) — để nếu train bị dừng/crash giữa
     chừng vẫn biết exp đó đang chạy kiến trúc nào, tiện khi test nhiều
     cấu trúc model khác nhau.
  8. export_all() được truyền thêm calib_loader/test_loader/export_tflite/
     n_calib/dataset_yaml — để tự động export TFLite int8, tự chạy so sánh
     accuracy 3 backend (PyTorch/ONNX/TFLite), và tự benchmark latency/FPS
     thật trên toàn bộ test_set (bench_dataset.py) ngay sau khi train xong.
"""

from __future__ import annotations

import random
import platform
import shutil
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
        model:        GLKANet
        cfg:          dict parse từ ccmt.yaml (toàn bộ)
        save_dir:     Path thư mục exp
        class_names:  list tên class từ dataloader
        resume_ckpt:  path tới 1 checkpoint (.pt) đã lưu trước đó
                      (best_f1.pt / best_loss.pt / bất kỳ file nào lưu
                      bởi _save_ckpt) để load lại weights + optimizer +
                      scheduler + scaler state và train tiếp từ đó.
                      cfg["train"]["epochs"] là TỔNG số epoch mong muốn
                      (không phải số epoch train thêm) — vd checkpoint
                      đang ở epoch 80, muốn train thêm 120 epoch nữa thì
                      set epochs: 200 trong yaml.
    """

    def __init__(
        self,
        model:        nn.Module,
        cfg:          dict,
        save_dir:     Path,
        class_names:  List[str],
        resume_ckpt:  str | Path | None = None,
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
        self.tflite_project_dir = self.exp_cfg.get("tflite_project_dir", "TFlite")
        self.test_dir           = self.exp_cfg.get("test_dir", None)


        norm_cfg = cfg.get("normalize", {})
        self.norm_mean = tuple(norm_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.norm_std  = tuple(norm_cfg.get("std",  [0.229, 0.224, 0.225]))
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

        self.hist       = dict(train_loss=[], val_loss=[], val_f1=[], val_acc=[])
        self.best_f1    = 0.0
        self.best_loss  = float("inf")
        self.start_epoch = 0

        # ── Resume ──────────────────────────────────────────────────
        if resume_ckpt is not None:
            self._resume(resume_ckpt)

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

        if self.use_amp:
            print("  [trainer] AMP (mixed precision) bật — fp16 trên CUDA")

        if resume_ckpt is not None:
            print(f"  [trainer] Resume: bắt đầu từ epoch {self.start_epoch + 1}/{self.epochs}")

    # ── Resume helper ────────────────────────────────────────────
    def _resume(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_ckpt không tồn tại: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["state_dict"])

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("  [trainer] ⚠ checkpoint không có optimizer state "
                  "(checkpoint cũ trước khi có resume) — chỉ resume weights, "
                  "momentum/LR sẽ khởi tạo lại từ đầu")

        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print("  [trainer] ⚠ checkpoint không có scheduler state — "
                  "LR sẽ tính lại từ epoch 0 theo T_max mới")

        if "scaler" in ckpt and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.start_epoch = ckpt.get("epoch", -1) + 1

        self.best_f1   = ckpt.get("best_f1",   ckpt.get("val_f1",   0.0))
        self.best_loss = ckpt.get("best_loss", ckpt.get("val_loss", float("inf")))

        if self.start_epoch >= self.epochs:
            print(f"  [trainer] ⚠ checkpoint đã ở epoch {self.start_epoch}, "
                  f">= epochs cấu hình ({self.epochs}). Tăng train.epochs "
                  f"trong yaml nếu muốn train thêm.")

    # ── Copy yaml kiến trúc helper ───────────────────────────────
    def _copy_arch_yaml(self, model_yaml: str | Path | None) -> None:
        """Copy file yaml kiến trúc model ra weights/ NGAY LÚC BẮT ĐẦU TRAIN."""
        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        if model_yaml is None:
            print("  [trainer] ⚠ không truyền model_yaml vào run() — sẽ KHÔNG lưu được "
                "yaml kiến trúc ngay từ đầu train.")
            return

        model_yaml = Path(model_yaml)
        if not model_yaml.exists():
            print(f"  [trainer] ⚠ model_yaml không tồn tại: {model_yaml} — bỏ qua copy yaml.")
            return

        dst = weights_dir / model_yaml.name
        shutil.copy2(model_yaml, dst)
        print(f"  [trainer] yaml kiến trúc → weights/{dst.name} "
            f"(copy ngay lúc bắt đầu train, trước khi vào training loop)")

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
        """Full train → val → test → export.

        Args:
            train_loader, val_loader, test_loader: DataLoader tương ứng.
            model_yaml:  đường dẫn yaml kiến trúc model.
            cfg_path:    đường dẫn yaml kiến trúc model — được copy vào
                         weights/ NGAY LÚC BẮT ĐẦU TRAIN, và dùng lại
                         trong export_all() để đóng gói cùng checkpoint.
            data_yaml:   đường dẫn dataset.yaml (path/train/test) — được
                         truyền thẳng xuống export_all() làm dataset_yaml
                         để tự động chạy bench_dataset.py (benchmark
                         latency/FPS/accuracy thật trên toàn bộ test_set)
                         sau khi export TFLite xong. Nếu không truyền, sẽ
                         thử lấy từ cfg["export"]["dataset_yaml"] trong
                         yaml cấu hình train.
        """
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        write_train_header(
            self.save_dir, self.cfg,
            len(self.class_names), total_params, self.class_names,
        )

        # ── Copy yaml kiến trúc NGAY TỪ ĐẦU, trước khi train ────────
        self._copy_arch_yaml(model_yaml)

        tsne_interval = self.log_cfg.get("tsne_interval", 20)
        log_every = 10  # chỉ .item() loss mỗi N step để giảm sync GPU-CPU

        # ══════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════
        for epoch in range(self.start_epoch, self.epochs):
            lr_curr = self.optimizer.param_groups[0]['lr']

            self.model.train()
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

                train_loss_sum += loss.detach() * imgs.size(0)
                n_seen += imgs.size(0)

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

            self._save_ckpt(
                "last.pt", epoch,
                val_f1=val_f1, val_loss=val_loss, val_acc=val_acc,
            )

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
                model         = self.model,
                save_dir      = self.save_dir,
                input_size    = self.cfg.get("img_size", 224),
                yaml_path     = model_yaml,
                opset         = self.exp_cfg.get("opset", 18),
                verbose       = True,
                export_tflite = self.exp_cfg.get("export_tflite", True),
                test_dir      = self.test_dir,
                dataset_yaml  = data_yaml or self.exp_cfg.get("dataset_yaml"),
                class_names   = self.class_names,
                tflite_project_dir = self.tflite_project_dir,
                tflite_mode   = self.exp_cfg.get("tflite_mode", "all"),
                n_calib       = self.exp_cfg.get("n_calib", 200),
                mean          = self.norm_mean,
                std           = self.norm_std,
                test_loader   = test_loader,
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
            {
                "state_dict": self.model.state_dict(),
                "epoch":      epoch,
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
                "scaler":     self.scaler.state_dict() if self.use_amp else None,
                "best_f1":    self.best_f1,
                "best_loss":  self.best_loss,
                **metrics,
            },
            self.save_dir / name,
        )

    def _load_ckpt(self, name: str) -> None:
        ckpt = torch.load(
            self.save_dir / name,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])