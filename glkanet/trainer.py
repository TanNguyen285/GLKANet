"""glkanet/trainer.py — Train loop, optimizer, scheduler, evaluate."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
# Optimizer + Scheduler factory
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
# Evaluate
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, list, list, list]:
    """Trả về (avg_loss, preds, labels, features)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_feats = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).long()
        logits, feats = model(imgs)
        total_loss   += criterion(logits, labels).item() * imgs.size(0)
        preds         = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_feats.extend(feats.cpu().numpy())

    return total_loss / len(loader.dataset), all_preds, all_labels, all_feats


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────

class Trainer:
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

        label_smoothing   = tr_cfg.get("label_smoothing", 0.0)
        self.criterion    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer    = build_optimizer(model, tr_cfg["optimizer"])
        self.scheduler    = build_scheduler(
            self.optimizer, tr_cfg["scheduler"], self.epochs)

        self.model.to(self.device)

        # History
        self.hist      = dict(train_loss=[], val_loss=[], val_f1=[], val_acc=[])
        self.best_f1   = 0.0
        self.best_loss = float("inf")

    # ── Run ───────────────────────────────────────────────────
    def run(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        model_yaml: str | Path | None = None,
        cfg_path:   str | Path | None = None,
    ) -> dict:
        """Full train → val → test → export.

        Returns:
            {"best_f1": float, "best_loss": float, "save_dir": Path}
        """
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        write_train_header(
            self.save_dir, self.cfg,
            len(self.class_names), total_params, self.class_names,
        )

        tsne_interval = self.log_cfg.get("tsne_interval", 20)

        # ══════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}]  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            # ── Train ──
            self.model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_loader.dataset)

            # ── Val ──
            val_loss, val_preds, val_labels, val_feats = evaluate(
                self.model, val_loader, self.criterion, self.device)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)

            self.hist["train_loss"].append(train_loss)
            self.hist["val_loss"].append(val_loss)
            self.hist["val_f1"].append(val_f1)
            self.hist["val_acc"].append(val_acc * 100)

            self.scheduler.step()

            print(f"  loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                  f"| acc={val_acc*100:.2f}% | f1={val_f1:.4f}")

            save_report(
                val_preds, val_labels, self.class_names, self.save_dir,
                tag="val", epoch=epoch,
                train_loss=train_loss, val_loss=val_loss, val_f1=val_f1,
            )

            # ── Checkpoint: best F1 ──
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self._save_ckpt("best_f1.pt", epoch, val_f1=val_f1, val_acc=val_acc)
                print(f"  [↑ F1] {self.best_f1:.4f}")
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
                print(f"  [↓ Loss] {self.best_loss:.4f}")

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
                self.model, test_loader, self.criterion, self.device)
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
                model=self.model,
                save_dir=self.save_dir,
                input_size=self.cfg["data"]["img_size"],
                yaml_path=cfg_path,
                opset=self.exp_cfg.get("opset", 18),
                verbose=True,
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