"""data/ccmt_loader.py — CCMT Augmented Dataset loader.

Nhận config dict từ train.py (đã parse từ ccmt.yaml).
Không có hard-coded constant — mọi thứ đều từ cfg.
"""

from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _normalize_class_name(name: str) -> str:
    """Bỏ số đuôi: 'healthy1' → 'healthy'."""
    return re.sub(r"\d+$", "", name).strip()


def _full_class_name(crop: str, cls: str) -> str:
    return f"{crop}_{cls}"


# ──────────────────────────────────────────────────────────────
# Scan dataset
# ──────────────────────────────────────────────────────────────

def scan_split(
    root: Path,
    split_name: str,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Duyệt <root>/<crop>/<split_name>/<class>/ → (samples, class_names).

    Returns:
        samples:     list of (img_path, class_idx)
        class_names: sorted list of "Crop_ClassName"
    """
    crop_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not crop_dirs:
        raise RuntimeError(f"Không tìm thấy crop folder nào trong {root}")

    raw: List[Tuple[str, str]] = []
    class_set: set = set()
    found_split = False

    for crop_dir in crop_dirs:
        split_dir = crop_dir / split_name
        if not split_dir.exists():
            continue
        found_split = True
        crop_name = crop_dir.name

        for cd in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            full_name = _full_class_name(crop_name, _normalize_class_name(cd.name))
            class_set.add(full_name)
            for f in cd.iterdir():
                if f.suffix in IMAGE_EXTS:
                    raw.append((str(f), full_name))

    if not found_split:
        raise RuntimeError(
            f"Không tìm thấy '{split_name}/' trong bất kỳ crop folder nào\n"
            f"  Root: {root}\n"
            f"  Cấu trúc cần: <root>/<crop>/{split_name}/<class>/"
        )
    if not raw:
        raise RuntimeError(
            f"Tìm thấy '{split_name}/' nhưng không có ảnh nào.")

    class_names  = sorted(class_set)
    cls_to_idx   = {c: i for i, c in enumerate(class_names)}
    samples      = [(p, cls_to_idx[c]) for p, c in raw]
    return samples, class_names


def verify_splits(train_names: List[str], test_names: List[str]) -> None:
    if train_names != test_names:
        only_tr = set(train_names) - set(test_names)
        only_te = set(test_names)  - set(train_names)
        msg = "Class mismatch train_set vs test_set!\n"
        if only_tr:
            msg += f"  Chỉ train: {sorted(only_tr)}\n"
        if only_te:
            msg += f"  Chỉ test : {sorted(only_te)}\n"
        raise RuntimeError(msg)


# ──────────────────────────────────────────────────────────────
# Stratified val split
# ──────────────────────────────────────────────────────────────

def stratified_val_split(
    samples: List[Tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List, List]:
    assert 0.0 < val_ratio < 1.0
    rng = random.Random(seed)
    by_class: Dict[int, List] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    train_s, val_s = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_s.extend(items[:n_val])
        train_s.extend(items[n_val:])
    return train_s, val_s


# ──────────────────────────────────────────────────────────────
# Transforms — đọc từ cfg dict
# ──────────────────────────────────────────────────────────────

def build_train_transform(data_cfg: dict) -> transforms.Compose:
    img_size = data_cfg["img_size"]
    scale    = tuple(data_cfg.get("crop_scale", [0.7, 1.0]))
    cj       = data_cfg.get("color_jitter", {})
    norm     = data_cfg.get("normalize", {})
    mean     = norm.get("mean", [0.485, 0.456, 0.406])
    std      = norm.get("std",  [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(p=data_cfg.get("flip_h", 0.5)),
        transforms.RandomVerticalFlip(p=data_cfg.get("flip_v", 0.3)),
        transforms.ColorJitter(**cj) if cj else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(data_cfg: dict) -> transforms.Compose:
    img_size = data_cfg["img_size"]
    norm     = data_cfg.get("normalize", {})
    mean     = norm.get("mean", [0.485, 0.456, 0.406])
    std      = norm.get("std",  [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class CCMTDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────
# Factory — entry point
# ──────────────────────────────────────────────────────────────

def get_data_loaders(
    data_cfg:  dict,
    hw_cfg:    dict,
    batch_size: int,
):
    """Build train/val/test DataLoader từ config dict.

    Args:
        data_cfg:   cfg["data"] từ ccmt.yaml
        hw_cfg:     cfg["hardware"] từ ccmt.yaml
        batch_size: cfg["train"]["batch_size"]

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    root = Path(data_cfg["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {root}\n"
            f"Sửa data.root trong configs/ccmt.yaml"
        )

    val_ratio        = data_cfg.get("val_ratio",         0.1)
    seed             = data_cfg.get("seed",              42)
    handle_imbalance = data_cfg.get("handle_imbalance",  False)
    num_workers      = hw_cfg.get("num_workers",         4)
    pin_memory       = hw_cfg.get("pin_memory",          True)

    # Scan
    all_train, train_class_names = scan_split(root, "train_set")
    test_samples, test_class_names = scan_split(root, "test_set")
    verify_splits(train_class_names, test_class_names)
    class_names = train_class_names

    # Val split
    train_samples, val_samples = stratified_val_split(all_train, val_ratio, seed)

    # Transforms
    train_tf = build_train_transform(data_cfg)
    val_tf   = build_val_transform(data_cfg)

    # Datasets
    train_ds = CCMTDataset(train_samples, train_tf)
    val_ds   = CCMTDataset(val_samples,   val_tf)
    test_ds  = CCMTDataset(test_samples,  val_tf)

    # Weighted sampler
    sampler = None
    if handle_imbalance:
        labels       = [s[1] for s in train_samples]
        class_counts = np.bincount(labels, minlength=len(class_names))
        ratio        = class_counts.max() / (class_counts.min() + 1e-8)
        if ratio > 1.5:
            weights = 1.0 / torch.tensor(
                class_counts[[s[1] for s in train_samples]], dtype=torch.float)
            sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
            print(f"[⚠] Imbalance ratio={ratio:.1f}× → WeightedRandomSampler ON")
        else:
            print(f"[✓] Classes cân bằng (ratio={ratio:.1f}×)")

    kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        drop_last=True, **kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,
        shuffle=False, **kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2,
        shuffle=False, **kw,
    )

    _print_summary(train_samples, val_samples, test_samples,
                   class_names, val_ratio, root)
    return train_loader, val_loader, test_loader, class_names


# ──────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────

def get_class_weights(
    class_names: List[str],
    train_samples: List[Tuple[str, int]],
) -> torch.Tensor:
    labels  = [s[1] for s in train_samples]
    counts  = np.bincount(labels, minlength=len(class_names)).astype(float)
    weights = counts.sum() / (len(class_names) * counts)
    return torch.tensor(weights, dtype=torch.float)


def _print_summary(train_s, val_s, test_s, class_names, val_ratio, root):
    n_tr, n_val, n_te = len(train_s), len(val_s), len(test_s)
    total = n_tr + n_val + n_te
    print(f"\n{'='*62}")
    print(f"[✓] CCMT Augmented Dataset")
    print(f"    Root   : {root}")
    print(f"    Classes: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"      [{i:2d}] {name}")
    print(f"    Total  : {total:7,d} images")
    print(f"    ├─ Train: {n_tr:6,d}  (train_set × {1-val_ratio:.0%})")
    print(f"    ├─ Val  : {n_val:6,d}  (train_set × {val_ratio:.0%} stratified)")
    print(f"    └─ Test : {n_te:6,d}  (test_set nguyên)")
    print(f"{'='*62}\n")
