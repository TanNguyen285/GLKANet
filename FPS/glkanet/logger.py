"""glkanet/logger.py — Plots, reports, t-SNE.

Tất cả side-effects (file I/O, matplotlib) nằm ở đây,
các module khác không import matplotlib trực tiếp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)


# ──────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────

def plot_curves(
    train_losses: list,
    val_losses:   list,
    val_f1s:      list,
    val_accs:     list,
    save_dir:     Path,
) -> None:
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ep, train_losses, label="Train", linewidth=2)
    axes[0].plot(ep, val_losses,   label="Val",   linewidth=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].plot(ep, val_f1s, color="steelblue", linewidth=2)
    axes[1].set_title("Val F1 (macro)"); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1")

    axes[2].plot(ep, val_accs, color="green", linewidth=2)
    axes[2].set_title("Val Accuracy (%)"); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    preds:       list,
    labels:      list,
    class_names: List[str],
    save_dir:    Path,
    tag:         str,
) -> None:
    cm    = confusion_matrix(labels, preds)
    short = [c[:14] for c in class_names]
    n     = len(class_names)
    sz    = max(10, n * 0.65)

    plt.figure(figsize=(sz, sz * 0.9))
    sns.heatmap(
        cm, annot=(n <= 30), fmt="d", cmap="Blues",
        xticklabels=short, yticklabels=short, linewidths=0.3,
    )
    plt.title(f"Confusion Matrix [{tag}]")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / f"cm_{tag}.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────
# t-SNE
# ──────────────────────────────────────────────────────────────

def plot_tsne(
    features:    list,
    labels:      list,
    class_names: List[str],
    save_dir:    Path,
    tag:         str,
    max_points:  int = 1000,
) -> None:
    try:
        feat = np.array(features)
        lbl  = np.array(labels)
        if len(feat) > max_points:
            idx  = np.random.choice(len(feat), max_points, replace=False)
            feat = feat[idx]; lbl = lbl[idx]

        print("  [t-SNE] fitting...")
        emb = TSNE(
            n_components=2, perplexity=30, random_state=42,
            init="pca", learning_rate="auto",
        ).fit_transform(feat)

        plt.figure(figsize=(12, 9))
        for i, name in enumerate(class_names):
            m = lbl == i
            plt.scatter(emb[m, 0], emb[m, 1], label=name[:14], alpha=0.6, s=30)
        plt.title(f"t-SNE [{tag}]")
        plt.legend(fontsize=6, ncol=3, loc="best")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"tsne_{tag}.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [!] t-SNE error: {e}")


# ──────────────────────────────────────────────────────────────
# Classification report
# ──────────────────────────────────────────────────────────────

def save_report(
    preds:       list,
    labels:      list,
    class_names: List[str],
    save_dir:    Path,
    tag:         str,
    epoch:       Optional[int]   = None,
    train_loss:  Optional[float] = None,
    val_loss:    Optional[float] = None,
    val_f1:      Optional[float] = None,
) -> tuple[float, float]:
    """Ghi report ra file. Trả về (accuracy, f1_macro)."""
    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wei = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        labels, preds, target_names=class_names,
        zero_division=0, digits=4,
    )

    if epoch is not None:
        path = save_dir / "epoch_reports.txt"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\nEPOCH {epoch+1}  [{tag}]\n{'='*80}\n")
            if train_loss is not None:
                f.write(f"Train Loss  : {train_loss:.4f}\n")
            if val_loss is not None:
                f.write(f"Val Loss    : {val_loss:.4f}\n")
            f.write(f"Accuracy    : {acc:.4f}\n")
            f.write(f"F1 macro    : {f1_mac:.4f}\n")
            f.write(f"F1 weighted : {f1_wei:.4f}\n\n{report}\n")
    else:
        path = save_dir / f"report_{tag}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy    : {acc:.4f}\n")
            f.write(f"F1 macro    : {f1_mac:.4f}\n")
            f.write(f"F1 weighted : {f1_wei:.4f}\n\n{report}\n")

    return acc, f1_mac


# ──────────────────────────────────────────────────────────────
# Train header
# ──────────────────────────────────────────────────────────────

def write_train_header(
    save_dir:     Path,
    cfg:          dict,
    n_classes:    int,
    total_params: float,
    class_names:  List[str],
) -> None:
    tr  = cfg["train"]
    opt = tr["optimizer"]
    sch = tr["scheduler"]
    with open(save_dir / "epoch_reports.txt", "w", encoding="utf-8") as f:
        f.write("Training Configuration\n")
        f.write(f"  Model      : {cfg.get('model_yaml', 'simple_glka.yaml')}\n")
        f.write(f"  Data yaml  : {cfg.get('data', 'N/A')}\n")
        f.write(f"  Params     : {total_params:.3f}M\n")
        f.write(f"  Classes    : {n_classes}\n")
        for i, c in enumerate(class_names):
            f.write(f"    [{i:2d}] {c}\n")
        f.write(f"  Epochs     : {tr['epochs']}\n")
        f.write(f"  Batch size : {tr['batch_size']}\n")
        f.write(f"  Optimizer  : {opt['type']}  lr={opt['lr']}\n")
        f.write(f"  Scheduler  : {sch['type']}\n")
        f.write(f"  Device     : {cfg['hardware']['device']}\n")
        f.write(f"{'='*80}\n")
