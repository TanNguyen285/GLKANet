"""glkanet/data/loader.py — Universal image classification dataloader.

Auto-detect cấu trúc folder và split strategy:

  Cấu trúc hỗ trợ:
  ┌─────────────────────────────────────────────────────────┐
  │ A) ImageFolder chuẩn                                    │
  │    root/train/cls/img  (+val? +test?)                   │
  │                                                         │
  │ B) CCMT-style (multi-level)                             │
  │    root/group/train_set/cls/img  (+test_set?)           │
  └─────────────────────────────────────────────────────────┘

  Split strategy (tự động):
  • train + val + test  → giữ nguyên cả 3
  • train + val         → giữ nguyên, không có test
  • train + test        → val tách 10% từ train (stratified)
  • train only          → tự chia 80/10/10

Config yaml tối thiểu:
    data:
      root: "path/to/dataset"
      img_size: 224
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp"}

# Các tên folder được nhận dạng tự động
_TRAIN_NAMES = {"train", "train_set", "training"}
_VAL_NAMES   = {"val", "valid", "validation", "val_set"}
_TEST_NAMES  = {"test", "test_set", "testing", "eval"}

Samples = List[Tuple[str, int]]   # [(img_path, class_idx), ...]


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, samples: Samples, transform=None):
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
# Transforms
# ──────────────────────────────────────────────────────────────

def build_train_transform(data_cfg: dict) -> transforms.Compose:
    img_size = data_cfg.get("img_size", 224)
    scale    = tuple(data_cfg.get("crop_scale", [0.7, 1.0]))
    cj       = data_cfg.get("color_jitter", {
        "brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05
    })
    norm = data_cfg.get("normalize", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std  = norm.get("std",  [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(p=data_cfg.get("flip_h", 0.5)),
        transforms.RandomVerticalFlip(p=data_cfg.get("flip_v", 0.3)),
        transforms.ColorJitter(**cj),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(data_cfg: dict) -> transforms.Compose:
    img_size = data_cfg.get("img_size", 224)
    norm = data_cfg.get("normalize", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std  = norm.get("std",  [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────────────────────
# Scanner — quét folder lấy (path, class_name)
# ──────────────────────────────────────────────────────────────

def _scan_class_dir(class_dir: Path, class_name: str) -> List[Tuple[str, str]]:
    """Quét đệ quy 1 class folder → list (img_path, class_name)."""
    results = []
    for f in class_dir.rglob("*"):
        if f.is_file() and f.suffix in IMAGE_EXTS:
            results.append((str(f), class_name))
    return results


def _scan_split_dir(split_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Quét split_dir/class_name/img → (raw_samples, class_names).

    raw_samples: [(img_path, class_name), ...]
    """
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise RuntimeError(f"Không tìm thấy class folder nào trong: {split_dir}")

    raw: List[Tuple[str, str]] = []
    class_names: List[str] = []

    for cd in class_dirs:
        name = cd.name
        class_names.append(name)
        raw.extend(_scan_class_dir(cd, name))

    if not raw:
        raise RuntimeError(f"Không có ảnh nào trong: {split_dir}")

    return raw, class_names


def _to_indexed(
    raw: List[Tuple[str, str]],
    class_to_idx: Dict[str, int],
) -> Samples:
    return [(p, class_to_idx[c]) for p, c in raw]


# ──────────────────────────────────────────────────────────────
# Auto-detect layout
# ──────────────────────────────────────────────────────────────

class _Layout:
    """Kết quả detect layout."""
    def __init__(self, style: str, train: Path,
                 val: Optional[Path], test: Optional[Path]):
        self.style = style   # "imagefolder" | "ccmt"
        self.train = train
        self.val   = val
        self.test  = test

    def __repr__(self):
        return (f"Layout(style={self.style}, "
                f"train={self.train.name}, "
                f"val={self.val.name if self.val else None}, "
                f"test={self.test.name if self.test else None})")


def _detect_layout(root: Path) -> _Layout:
    """Auto-detect cấu trúc folder dataset.

    Ưu tiên:
    1. ImageFolder: root có chứa train/ (hoặc tên tương đương)
    2. CCMT-style: root/group/train_set/class/img
    """
    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}

    # ── Thử ImageFolder ──────────────────────────────────────
    train_dir = _find_split_dir(subdirs, _TRAIN_NAMES)
    if train_dir is not None:
        val_dir  = _find_split_dir(subdirs, _VAL_NAMES)
        test_dir = _find_split_dir(subdirs, _TEST_NAMES)
        return _Layout("imagefolder", train_dir, val_dir, test_dir)

    # ── Thử CCMT-style: root/group/train_set/class/ ──────────
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        inner = {d.name.lower(): d for d in group_dir.iterdir() if d.is_dir()}
        train_dir = _find_split_dir(inner, _TRAIN_NAMES)
        if train_dir is not None:
            val_dir  = _find_split_dir(inner, _VAL_NAMES)
            test_dir = _find_split_dir(inner, _TEST_NAMES)
            # CCMT: cần quét tất cả group → trả về root với style=ccmt
            return _Layout("ccmt", root, None, None)

    raise RuntimeError(
        f"Không nhận dạng được cấu trúc dataset tại: {root}\n"
        f"Hỗ trợ:\n"
        f"  ImageFolder : root/train/class/img.jpg\n"
        f"  CCMT-style  : root/group/train_set/class/img.jpg\n"
        f"Các thư mục con tìm thấy: {[d.name for d in root.iterdir() if d.is_dir()]}"
    )


def _find_split_dir(
    subdirs: Dict[str, Path],
    names: set,
) -> Optional[Path]:
    for name in names:
        if name in subdirs:
            return subdirs[name]
    return None


# ──────────────────────────────────────────────────────────────
# CCMT scanner (multi-group)
# ──────────────────────────────────────────────────────────────

def _scan_ccmt(root: Path) -> Tuple[Samples, Optional[Samples], Optional[Samples], List[str]]:
    """Quét CCMT-style: root/group/train_set|test_set/class/img."""
    train_raw: List[Tuple[str, str]] = []
    val_raw:   List[Tuple[str, str]] = []
    test_raw:  List[Tuple[str, str]] = []
    all_class_names: set = set()

    has_val  = False
    has_test = False

    for group_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        inner = {d.name.lower(): d for d in group_dir.iterdir() if d.is_dir()}
        prefix = group_dir.name   # dùng làm prefix cho class name

        train_d = _find_split_dir(inner, _TRAIN_NAMES)
        val_d   = _find_split_dir(inner, _VAL_NAMES)
        test_d  = _find_split_dir(inner, _TEST_NAMES)

        if train_d is None:
            continue

        # Quét từng split, gắn prefix vào class name
        for class_dir in sorted(d for d in train_d.iterdir() if d.is_dir()):
            full_name = f"{prefix}_{class_dir.name}"
            all_class_names.add(full_name)
            train_raw.extend(_scan_class_dir(class_dir, full_name))

        if val_d:
            has_val = True
            for class_dir in sorted(d for d in val_d.iterdir() if d.is_dir()):
                full_name = f"{prefix}_{class_dir.name}"
                val_raw.extend(_scan_class_dir(class_dir, full_name))

        if test_d:
            has_test = True
            for class_dir in sorted(d for d in test_d.iterdir() if d.is_dir()):
                full_name = f"{prefix}_{class_dir.name}"
                test_raw.extend(_scan_class_dir(class_dir, full_name))

    class_names  = sorted(all_class_names)
    cls_to_idx   = {c: i for i, c in enumerate(class_names)}

    train_s = _to_indexed(train_raw, cls_to_idx)
    val_s   = _to_indexed(val_raw,   cls_to_idx) if has_val  else None
    test_s  = _to_indexed(test_raw,  cls_to_idx) if has_test else None

    return train_s, val_s, test_s, class_names


# ──────────────────────────────────────────────────────────────
# Split strategies
# ──────────────────────────────────────────────────────────────

def _stratified_split(
    samples: Samples,
    ratio:   float,
    seed:    int,
) -> Tuple[Samples, Samples]:
    """Tách `ratio` phần từ samples (stratified theo class)."""
    rng = random.Random(seed)
    by_class: Dict[int, Samples] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    a, b = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n = max(1, int(len(items) * ratio))
        b.extend(items[:n])
        a.extend(items[n:])
    return a, b


def _apply_split_strategy(
    train_s: Samples,
    val_s:   Optional[Samples],
    test_s:  Optional[Samples],
    val_ratio:  float,
    test_ratio: float,
    seed: int,
) -> Tuple[Samples, Samples, Optional[Samples]]:
    """Áp dụng split strategy tự động.

    Returns:
        (train, val, test)  — test có thể là None
    """
    has_val  = val_s  is not None and len(val_s)  > 0
    has_test = test_s is not None and len(test_s) > 0

    if has_val and has_test:
        # Case 1: train + val + test → giữ nguyên cả 3
        strategy = "fixed train/val/test"
        final_train, final_val, final_test = train_s, val_s, test_s

    elif has_val and not has_test:
        # Case 2: train + val → giữ nguyên, không có test
        strategy = "fixed train/val (no test)"
        final_train, final_val, final_test = train_s, val_s, None

    elif not has_val and has_test:
        # Case 3: train + test → val tách 10% từ train
        strategy = f"auto val {val_ratio:.0%} from train, fixed test"
        final_train, final_val = _stratified_split(train_s, val_ratio, seed)
        final_test = test_s

    else:
        # Case 4: train only → tự chia 80/10/10
        strategy = f"auto split {1-val_ratio-test_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}"
        tmp, final_test = _stratified_split(train_s, test_ratio, seed)
        final_train, final_val = _stratified_split(tmp, val_ratio / (1 - test_ratio), seed)

    print(f"  [split] {strategy}")
    return final_train, final_val, final_test


# ──────────────────────────────────────────────────────────────
# Weighted sampler
# ──────────────────────────────────────────────────────────────

def _make_sampler(
    train_samples: Samples,
    n_classes: int,
) -> Optional[WeightedRandomSampler]:
    labels       = [s[1] for s in train_samples]
    class_counts = np.bincount(labels, minlength=n_classes).astype(float)
    ratio        = class_counts.max() / (class_counts.min() + 1e-8)
    if ratio > 1.5:
        weights = 1.0 / torch.tensor(class_counts[labels], dtype=torch.float)
        print(f"  [sampler] imbalance={ratio:.1f}× → WeightedRandomSampler ON")
        return WeightedRandomSampler(weights, len(train_samples), replacement=True)
    print(f"  [sampler] balanced (ratio={ratio:.1f}×)")
    return None


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def _print_summary(
    root:        Path,
    layout:      _Layout,
    class_names: List[str],
    train_s:     Samples,
    val_s:       Samples,
    test_s:      Optional[Samples],
) -> None:
    n_tr  = len(train_s)
    n_val = len(val_s)
    n_te  = len(test_s) if test_s else 0
    total = n_tr + n_val + n_te

    print(f"\n{'='*62}")
    print(f"  Dataset  : {root.name}")
    print(f"  Layout   : {layout.style}")
    print(f"  Classes  : {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"    [{i:2d}] {name}")
    print(f"  Total    : {total:,} images")
    print(f"  ├─ Train : {n_tr:,}")
    print(f"  ├─ Val   : {n_val:,}")
    print(f"  └─ Test  : {n_te:,}" + (" (N/A)" if n_te == 0 else ""))
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────
# Main factory
# ──────────────────────────────────────────────────────────────

def get_data_loaders(
    data_cfg:   dict,
    hw_cfg:     dict,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """Build train/val/test DataLoader tự động từ config dict.

    Nhận vào:
        data_cfg:   cfg["data"] từ yaml
        hw_cfg:     cfg["hardware"] từ yaml
        batch_size: cfg["train"]["batch_size"]

    Trả về:
        train_loader, val_loader, test_loader (hoặc None), class_names
    """
    root = Path(data_cfg["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {root}\n"
            f"Kiểm tra lại data.root trong config yaml."
        )

    val_ratio        = data_cfg.get("val_ratio",         0.1)
    test_ratio       = data_cfg.get("test_ratio",        0.1)
    seed             = data_cfg.get("seed",              42)
    handle_imbalance = data_cfg.get("handle_imbalance",  False)
    num_workers      = hw_cfg.get("num_workers",         4)
    pin_memory       = hw_cfg.get("pin_memory",          True)

    # ── 1. Detect layout ──────────────────────────────────────
    layout = _detect_layout(root)
    print(f"\n  [loader] Detected: {layout}")

    # ── 2. Scan raw samples ───────────────────────────────────
    if layout.style == "ccmt":
        train_s, val_s, test_s, class_names = _scan_ccmt(root)
    else:
        # ImageFolder
        train_raw, train_cls = _scan_split_dir(layout.train)

        val_raw,  val_cls  = (None, None)
        test_raw, test_cls = (None, None)

        if layout.val:
            val_raw, val_cls = _scan_split_dir(layout.val)
        if layout.test:
            test_raw, test_cls = _scan_split_dir(layout.test)

        # Hợp nhất class names từ tất cả split
        all_cls = set(train_cls)
        if val_cls:  all_cls |= set(val_cls)
        if test_cls: all_cls |= set(test_cls)
        class_names = sorted(all_cls)
        cls_to_idx  = {c: i for i, c in enumerate(class_names)}

        train_s = _to_indexed(train_raw, cls_to_idx)
        val_s   = _to_indexed(val_raw,   cls_to_idx) if val_raw  else None
        test_s  = _to_indexed(test_raw,  cls_to_idx) if test_raw else None

    # ── 3. Auto split strategy ────────────────────────────────
    train_s, val_s, test_s = _apply_split_strategy(
        train_s, val_s, test_s,
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
    )

    # ── 4. Transforms ─────────────────────────────────────────
    train_tf = build_train_transform(data_cfg)
    val_tf   = build_val_transform(data_cfg)

    # ── 5. Datasets ───────────────────────────────────────────
    train_ds = ImageDataset(train_s, train_tf)
    val_ds   = ImageDataset(val_s,   val_tf)
    test_ds  = ImageDataset(test_s,  val_tf) if test_s else None

    # ── 6. Sampler ────────────────────────────────────────────
    sampler = _make_sampler(train_s, len(class_names)) \
              if handle_imbalance else None

    # ── 7. DataLoaders ────────────────────────────────────────
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
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **kw)
        if test_ds else None
    )

    _print_summary(root, layout, class_names, train_s, val_s, test_s)
    return train_loader, val_loader, test_loader, class_names