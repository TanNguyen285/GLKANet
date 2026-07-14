from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp"}
Samples    = List[Tuple[str, int]]   # [(img_path, class_idx), ...]


# ──────────────────────────────────────────────────────────────
# Data YAML parser  (KHÔNG ĐỔI LOGIC — giữ nguyên 100% so với bản gốc)
# ──────────────────────────────────────────────────────────────

class DataConfig:
    """Parse data yaml → resolved paths."""

    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path)
        raw = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))

        root_raw = raw.get("path", None)
        if root_raw:
            root = Path(root_raw)
            if not root.is_absolute():
                root = (self.yaml_path.parent / root).resolve()
        else:
            root = self.yaml_path.parent
        self.root = root

        self.train_dir = self._resolve(raw.get("train"), root)
        self.val_dir   = self._resolve(raw.get("val"),   root)
        self.test_dir  = self._resolve(raw.get("test"),  root)

        if self.train_dir is None:
            self.train_dir = root

        self.nc:    Optional[int]       = raw.get("nc",    None)
        self.names: Optional[List[str]] = raw.get("names", None)

    @staticmethod
    def _resolve(val, root: Path) -> Optional[Path]:
        if val is None or str(val).strip() == "":
            return None
        p = Path(str(val))
        if p.is_absolute():
            return p
        return (root / p).resolve()

    def __repr__(self):
        return (
            f"DataConfig(\n"
            f"  train={self.train_dir}\n"
            f"  val  ={self.val_dir}\n"
            f"  test ={self.test_dir}\n"
            f"  nc={self.nc}  names={self.names}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# Helpers (giữ nguyên)
# ──────────────────────────────────────────────────────────────

def _normalize_class_name(raw: str) -> str:
    return re.sub(r'\d+$', '', raw).strip()


def _has_images(directory: Path) -> bool:
    return any(
        f.suffix in IMAGE_EXTS
        for f in directory.iterdir()
        if f.is_file()
    )


def _detect_structure(candidate: Path) -> str:
    if not candidate.exists():
        return "missing"

    subdirs = [d for d in candidate.iterdir() if d.is_dir()]
    if not subdirs:
        return "missing"

    first = subdirs[0]
    first_subs = [d for d in first.iterdir() if d.is_dir()]

    if not first_subs:
        return "split"

    # Tiny-ImageNet kiểu: train/<classdir>/images/*.JPEG
    # classdir chỉ có ĐÚNG 1 thư mục con duy nhất tên "images" chứa ảnh
    # phẳng (không phải nhiều "class con" như CCMT group/class/*.jpg).
    # → đây vẫn là 1 class = classdir, KHÔNG phải group/class hai tầng.
    if len(first_subs) == 1 and first_subs[0].name.lower() == "images" \
            and _has_images(first_subs[0]):
        return "split"

    first_sub_sub = first_subs[0]
    sub_sub_subs = [d for d in first_sub_sub.iterdir() if d.is_dir()]

    if not sub_sub_subs:
        if _has_images(first_sub_sub):
            return "flat"
        if _has_images(first_subs[0]):
            return "split"
        return "flat"

    return "ccmt"


# ──────────────────────────────────────────────────────────────
# Scanners
# ──────────────────────────────────────────────────────────────

def _scan_split_dir(split_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy: {split_dir}")

    raw:         List[Tuple[str, str]] = []
    class_names: set                   = set()

    for cd in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        # FIX: KHONG normalize o day. _normalize_class_name xoa het chu
        # so cuoi chuoi, ma ten class kieu ImageFolder/Tiny-ImageNet
        # (vd "n01443537") toan la so sau chu dau -> bi xoa con "n",
        # gay collapse tat ca class thanh 1 ("n"), lech mapping so voi
        # val (val dung _scan_val_annotations, khong normalize).
        # Normalize chi hop ly cho CCMT (ten kieu "healthy1" -> "healthy"),
        # duoc xu ly rieng trong _scan_ccmt_split / _scan_flat_ccmt.
        name = cd.name
        class_names.add(name)
        for f in cd.rglob("*"):
            if f.is_file() and f.suffix in IMAGE_EXTS:
                raw.append((str(f), name))

    if not raw:
        raise RuntimeError(f"Không có ảnh trong: {split_dir}")
    return raw, sorted(class_names)


def _scan_ccmt_split(
    root: Path,
    split_name: str,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    raw:         List[Tuple[str, str]] = []
    class_names: set                   = set()

    group_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    found = False
    for group_dir in group_dirs:
        split_dir = group_dir / split_name
        if not split_dir.exists():
            continue
        found = True
        group_name = group_dir.name
        for cd in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            cls = _normalize_class_name(cd.name)
            full_name = f"{group_name}_{cls}"
            class_names.add(full_name)
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix in IMAGE_EXTS:
                    raw.append((str(f), full_name))

    if not found:
        raise RuntimeError(
            f"Không tìm thấy '{split_name}/' trong bất kỳ group nào tại: {root}"
        )
    if not raw:
        raise RuntimeError(f"Không có ảnh trong CCMT split '{split_name}'")

    return raw, sorted(class_names)


def _scan_flat_ccmt(
    root: Path,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    raw:         List[Tuple[str, str]] = []
    class_names: set                   = set()

    group_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not group_dirs:
        raise RuntimeError(f"Không có group folder nào trong: {root}")

    for group_dir in group_dirs:
        group_name = group_dir.name
        class_dirs = [d for d in group_dir.iterdir() if d.is_dir()]
        for cd in sorted(class_dirs):
            cls       = _normalize_class_name(cd.name)
            full_name = f"{group_name}_{cls}"
            class_names.add(full_name)
            for f in cd.rglob("*"):
                if f.is_file() and f.suffix in IMAGE_EXTS:
                    raw.append((str(f), full_name))

    if not raw:
        raise RuntimeError(f"Không có ảnh nào trong flat CCMT: {root}")
    return raw, sorted(class_names)


# ──────────────────────────────────────────────────────────────
# Val annotations parser (Tiny-ImageNet gốc: val/images/*.JPEG +
# val_annotations.txt, không có class-subdir sẵn)
# ──────────────────────────────────────────────────────────────

def _scan_val_annotations(val_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse val/val_annotations.txt (Tiny-ImageNet gốc):
    filename<TAB>class_id<TAB>xmin<TAB>ymin<TAB>xmax<TAB>ymax

    Ảnh có thể nằm ở nhiều dạng cấu trúc khác nhau tuỳ cách giải nén:
      - phẳng trong val/images/*.JPEG (cấu trúc gốc)
      - phẳng trực tiếp trong val/*.JPEG
      - đã bị sắp xếp lại theo class-subdir: val/<class_id>/.../*.JPEG
        (một số script tự động tổ chức lại Tiny-ImageNet val theo ImageFolder)
    → build sẵn 1 map {tên_file: full_path} bằng cách quét đệ quy toàn bộ
    val_dir MỘT LẦN, rồi tra cứu theo tên thay vì đoán đường dẫn cố định.
    """
    ann_path = val_dir / "val_annotations.txt"
    if not ann_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {ann_path}")

    # Quét đệ quy 1 lần, build map tên file -> full path (không quan tâm
    # nằm ở subfolder nào, khớp mọi kiểu tổ chức thư mục val/ khác nhau)
    name_to_path: Dict[str, str] = {}
    for f in val_dir.rglob("*"):
        if f.is_file() and f.suffix in IMAGE_EXTS:
            name_to_path[f.name] = str(f)

    raw:         List[Tuple[str, str]] = []
    class_names: set                   = set()

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fname, cls = parts[0], parts[1]
            img_path = name_to_path.get(fname)
            if img_path is None:
                continue
            class_names.add(cls)
            raw.append((img_path, cls))

    if not raw:
        raise RuntimeError(f"Không có ảnh nào khớp với {ann_path}")
    return raw, sorted(class_names)


# ──────────────────────────────────────────────────────────────
# Unlabeled split detection (test set không nhãn, kiểu
# test/images/*.JPEG, không có class-subdir)
# ──────────────────────────────────────────────────────────────

def _is_labeled_split(directory: Path) -> bool:
    """
    True nếu directory có class-subdir (train kiểu Tiny-ImageNet:
    train/n01443537/images/xxx.JPEG). False nếu chỉ chứa ảnh phẳng
    (trực tiếp hoặc qua đúng 1 thư mục "images/" duy nhất) — đây là
    trường hợp test set không có nhãn. (val dùng _scan_val_annotations
    riêng, không đi qua hàm này).
    """
    if not directory.exists():
        return False
    if _has_images(directory):
        return False
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.lower() == "images":
        return False
    return True


def _scan_unlabeled_dir(directory: Path) -> List[str]:
    """Quét ảnh đệ quy, không nhãn. Dùng cho test/images/xxx.JPEG."""
    if not directory.exists():
        raise FileNotFoundError(f"Không tìm thấy: {directory}")
    paths = sorted(
        str(f) for f in directory.rglob("*")
        if f.is_file() and f.suffix in IMAGE_EXTS
    )
    if not paths:
        raise RuntimeError(f"Không có ảnh trong: {directory}")
    return paths


def _to_indexed(
    raw: List[Tuple[str, str]],
    cls_to_idx: Dict[str, int],
) -> Samples:
    out = []
    for p, c in raw:
        if c not in cls_to_idx:
            continue
        out.append((p, cls_to_idx[c]))
    return out


# ──────────────────────────────────────────────────────────────
# Stratified split (giữ nguyên — chỉ dùng khi yaml THIẾU val/test)
# ──────────────────────────────────────────────────────────────

def _stratified_split(
    samples: Samples,
    ratio:   float,
    seed:    int,
) -> Tuple[Samples, Samples]:
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
    train_s:    Samples,
    val_s:      Optional[Samples],
    test_s:     Optional[Samples],
    val_ratio:  float,
    test_ratio: float,
    seed:       int,
) -> Tuple[Samples, Samples, Optional[Samples]]:
    has_val  = bool(val_s)
    has_test = bool(test_s)

    if has_val and has_test:
        return train_s, val_s, test_s
    if has_val and not has_test:
        return train_s, val_s, None
    if not has_val and has_test:
        tr, vl = _stratified_split(train_s, val_ratio, seed)
        return tr, vl, test_s
    tmp, te = _stratified_split(train_s, test_ratio, seed)
    tr, vl  = _stratified_split(tmp, val_ratio / (1 - test_ratio + 1e-9), seed)
    return tr, vl, te


# ──────────────────────────────────────────────────────────────
# Dataset + Transforms
# ──────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    - Không print() trong __getitem__ → tránh log spam (Windows console I/O chậm).
    - Image.draft() decode nhanh hơn cho JPEG khi ảnh gốc lớn hơn target nhiều.
    - Resize ảnh về kích thước cố định TRƯỚC khi đưa vào augment.
    """

    def __init__(
        self,
        samples:     Samples,
        transform=None,
        target_size: int = 224,
        pre_resize:  int = 0,
    ):
        self.samples     = samples
        self.transform   = transform
        self.target_size = target_size
        self.pre_resize  = pre_resize or int(round(target_size * 1.15))
        self._err_count  = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            try:
                img.draft("RGB", (self.pre_resize, self.pre_resize))
            except Exception:
                pass
            img = img.convert("RGB")

            w, h = img.size
            if w != h:
                scale = self.pre_resize / min(w, h)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
                left = (nw - self.pre_resize) // 2
                top  = (nh - self.pre_resize) // 2
                img = img.crop((left, top, left + self.pre_resize, top + self.pre_resize))
            elif w != self.pre_resize:
                img = img.resize((self.pre_resize, self.pre_resize), Image.BILINEAR)

            return img
        except Exception:
            self._err_count += 1
            return None

    def __getitem__(self, idx: int):
        for offset in range(5):
            path, label = self.samples[(idx + offset) % len(self.samples)]
            img = self._load(path)
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                return img, label
        img = torch.zeros(3, self.target_size, self.target_size)
        return img, label

    def error_summary(self) -> int:
        return self._err_count


class UnlabeledImageDataset(Dataset):
    """Giống ImageDataset nhưng cho test set KHÔNG có nhãn.
    Trả về (img, filename) thay vì (img, label)."""

    def __init__(
        self,
        paths:       List[str],
        transform=None,
        target_size: int = 224,
        pre_resize:  int = 0,
    ):
        self.paths       = paths
        self.transform   = transform
        self.target_size = target_size
        self.pre_resize  = pre_resize or int(round(target_size * 1.15))
        self._err_count  = 0

    def __len__(self) -> int:
        return len(self.paths)

    # Tái dùng nguyên logic decode/resize của ImageDataset
    _load = ImageDataset._load

    def __getitem__(self, idx: int):
        for offset in range(5):
            path = self.paths[(idx + offset) % len(self.paths)]
            img = self._load(path)
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                return img, Path(path).name
        img = torch.zeros(3, self.target_size, self.target_size)
        return img, Path(path).name

    def error_summary(self) -> int:
        return self._err_count


def build_train_transform(train_cfg: dict) -> transforms.Compose:
    aug      = train_cfg.get("augment", {})
    norm     = train_cfg.get("normalize", {})
    img_size = train_cfg.get("img_size", 224)
    mean     = norm.get("mean", [0.485, 0.456, 0.406])
    std      = norm.get("std",  [0.229, 0.224, 0.225])
    cj       = aug.get("color_jitter", {
        "brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05
    })
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=tuple(aug.get("crop_scale", [0.7, 1.0]))),
        transforms.RandomHorizontalFlip(p=aug.get("flip_h", 0.5)),
        transforms.RandomVerticalFlip(p=aug.get("flip_v", 0.3)),
        transforms.ColorJitter(**cj),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(train_cfg: dict) -> transforms.Compose:
    norm     = train_cfg.get("normalize", {})
    img_size = train_cfg.get("img_size", 224)
    mean     = norm.get("mean", [0.485, 0.456, 0.406])
    std      = norm.get("std",  [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────────────────────
# Summary (giữ nguyên)
# ──────────────────────────────────────────────────────────────

def _print_summary(
    dcfg:        DataConfig,
    class_names: List[str],
    train_s:     Samples,
    val_s:       Samples,
    test_s:      Optional[Samples],
    strategy:    str,
    n_test_unlabeled: int = 0,
) -> None:
    n_tr  = len(train_s)
    n_val = len(val_s)
    n_te  = len(test_s) if test_s else 0
    print(f"\n{'='*62}")
    print(f"  Data yaml : {dcfg.yaml_path.name}")
    print(f"  Strategy  : {strategy}")
    print(f"  Classes   : {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"    [{i:2d}] {name}")
    print(f"  Images    : {n_tr+n_val+n_te:,} total (chưa tính test không nhãn)")
    print(f"    train   : {n_tr:,}")
    print(f"    val     : {n_val:,}")
    print(f"    test    : {n_te:,}" + (" (N/A)" if n_te == 0 else ""))
    if n_test_unlabeled:
        print(f"    test (không nhãn): {n_test_unlabeled:,}  → test_loader trả (img, filename)")
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────
# Main factory
# ──────────────────────────────────────────────────────────────

def _find_ccmt_split_name(root: Path, prefer: str = "train") -> str:
    first_group = next((d for d in root.iterdir() if d.is_dir()), None)
    if first_group is None:
        raise RuntimeError(f"Không có group folder nào trong: {root}")

    candidates = [d.name for d in first_group.iterdir() if d.is_dir()]
    if not candidates:
        raise RuntimeError(f"Group '{first_group.name}' không có thư mục split nào bên trong.")

    match = next((c for c in candidates if re.search(prefer, c, re.IGNORECASE)), None)
    if match:
        return match
    return candidates[0]


def get_data_loaders(
    data_yaml:  str | Path,
    train_cfg:  dict,
    hw_cfg:     dict,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    dcfg = DataConfig(data_yaml)

    val_ratio   = train_cfg.get("val_ratio",  0.2)
    test_ratio  = train_cfg.get("test_ratio", 0.1)
    seed        = train_cfg.get("seed",       42)
    imbalance   = train_cfg.get("handle_imbalance", False)
    img_size    = train_cfg.get("img_size", 224)

    num_workers_train = hw_cfg.get("num_workers", 8)
    num_workers_eval  = hw_cfg.get("num_workers_eval", min(2, num_workers_train))
    pin_memory        = hw_cfg.get("pin_memory", True)
    prefetch_train    = hw_cfg.get("prefetch_factor", 2) if num_workers_train > 0 else None
    prefetch_eval     = 2 if num_workers_eval > 0 else None

    struct = _detect_structure(dcfg.train_dir)

    if struct == "missing" and dcfg.root.exists():
        split_name = dcfg.train_dir.name
        struct     = "ccmt"
    elif struct == "ccmt" and dcfg.train_dir == dcfg.root:
        split_name = _find_ccmt_split_name(dcfg.root, prefer="train")
        print(f"  [loader] yaml không khai báo 'train:' → tự dò split: '{split_name}'")
    else:
        split_name = dcfg.train_dir.name

    test_unlabeled_paths: Optional[List[str]] = None

    if struct == "flat":
        strategy = f"CCMT Gốc — auto {1-val_ratio-test_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}"
        print(f"  [loader] Phát hiện CCMT Gốc (flat) tại: {dcfg.train_dir}")

        all_raw, all_cls = _scan_flat_ccmt(dcfg.train_dir)
        class_names = list(dcfg.names) if dcfg.names else all_cls
        cls_to_idx  = {c: i for i, c in enumerate(class_names)}
        all_s       = _to_indexed(all_raw, cls_to_idx)

        train_s, val_s, test_s = _apply_split_strategy(
            all_s, None, None, val_ratio, test_ratio, seed)

    elif struct == "ccmt":
        strategy = "CCMT Augmented — fixed train/test"
        print(f"  [loader] Phát hiện CCMT Augmented tại: {dcfg.root}")

        train_raw, tr_cls = _scan_ccmt_split(dcfg.root, split_name)

        val_split_name = dcfg.val_dir.name if dcfg.val_dir and dcfg.val_dir != dcfg.root else None
        test_split_name = dcfg.test_dir.name if dcfg.test_dir and dcfg.test_dir != dcfg.root else None

        if test_split_name is None:
            try:
                test_split_name = _find_ccmt_split_name(dcfg.root, prefer="test")
            except RuntimeError:
                test_split_name = None

        val_raw, vl_cls = (
            _scan_ccmt_split(dcfg.root, val_split_name)
            if val_split_name else (None, [])
        )
        test_raw, te_cls = (
            _scan_ccmt_split(dcfg.root, test_split_name)
            if test_split_name else (None, [])
        )

        all_cls = set(tr_cls)
        if vl_cls: all_cls |= set(vl_cls)
        if te_cls: all_cls |= set(te_cls)

        class_names = list(dcfg.names) if dcfg.names else sorted(all_cls)
        cls_to_idx  = {c: i for i, c in enumerate(class_names)}
        train_s     = _to_indexed(train_raw, cls_to_idx)
        val_s       = _to_indexed(val_raw,  cls_to_idx) if val_raw  else None
        test_s      = _to_indexed(test_raw, cls_to_idx) if test_raw else None

        train_s, val_s, test_s = _apply_split_strategy(
            train_s, val_s, test_s, val_ratio, test_ratio, seed)

    else:
        strategy = "ImageFolder chuẩn"
        print(f"  [loader] Phát hiện ImageFolder tại: {dcfg.train_dir}")

        train_raw, tr_cls = _scan_split_dir(dcfg.train_dir)

        # val: ưu tiên val_annotations.txt (Tiny-ImageNet gốc: val/images/*.JPEG
        # phẳng, không có class-subdir), fallback về class-subdir nếu đã có sẵn.
        val_raw, vl_cls = None, []
        if dcfg.val_dir:
            if (dcfg.val_dir / "val_annotations.txt").exists():
                print(f"  [loader] val dùng val_annotations.txt tại: {dcfg.val_dir}")
                val_raw, vl_cls = _scan_val_annotations(dcfg.val_dir)
            else:
                val_raw, vl_cls = _scan_split_dir(dcfg.val_dir)

        # test: có thể không có nhãn (Tiny-ImageNet test/images/*.JPEG)
        test_raw, te_cls = None, []
        if dcfg.test_dir:
            if _is_labeled_split(dcfg.test_dir):
                test_raw, te_cls = _scan_split_dir(dcfg.test_dir)
            else:
                test_unlabeled_paths = _scan_unlabeled_dir(dcfg.test_dir)
                print(f"  [loader] test không có nhãn ({len(test_unlabeled_paths)} ảnh) "
                      f"tại: {dcfg.test_dir} → test_loader riêng (img, filename), "
                      f"không tham gia đánh giá bằng accuracy/F1.")

        all_cls = set(tr_cls)
        if vl_cls: all_cls |= set(vl_cls)
        if te_cls: all_cls |= set(te_cls)

        class_names = list(dcfg.names) if dcfg.names else sorted(all_cls)
        cls_to_idx  = {c: i for i, c in enumerate(class_names)}
        train_s     = _to_indexed(train_raw, cls_to_idx)
        val_s       = _to_indexed(val_raw,  cls_to_idx) if val_raw  else None
        test_s      = _to_indexed(test_raw, cls_to_idx) if test_raw else None

        train_s, val_s, test_s = _apply_split_strategy(
            train_s, val_s, test_s, val_ratio, test_ratio, seed)

    if dcfg.nc is not None and dcfg.nc != len(class_names):
        print(f"  [⚠] data yaml nc={dcfg.nc} nhưng scan ra {len(class_names)} class "
              f"→ dùng {len(class_names)}")

    train_tf = build_train_transform(train_cfg)
    val_tf   = build_val_transform(train_cfg)

    train_ds = ImageDataset(train_s, train_tf, target_size=img_size)
    val_ds   = ImageDataset(val_s,   val_tf,   target_size=img_size) if val_s else None
    test_ds  = ImageDataset(test_s,  val_tf,   target_size=img_size) if test_s else None

    test_unlabeled_ds = (
        UnlabeledImageDataset(test_unlabeled_paths, val_tf, target_size=img_size)
        if test_unlabeled_paths else None
    )

    sampler = None
    if imbalance:
        labels = [s[1] for s in train_s]
        counts = np.bincount(labels, minlength=len(class_names)).astype(float)
        ratio  = counts.max() / (counts.min() + 1e-8)
        if ratio > 1.5:
            w       = 1.0 / torch.tensor(counts[labels], dtype=torch.float)
            sampler = WeightedRandomSampler(w, len(train_ds), replacement=True)
            print(f"  [sampler] imbalance={ratio:.1f}× → WeightedRandomSampler ON")

    train_kw = dict(
        num_workers=num_workers_train,
        pin_memory=pin_memory,
        persistent_workers=(num_workers_train > 0),
    )
    if prefetch_train is not None:
        train_kw["prefetch_factor"] = prefetch_train

    eval_kw = dict(
        num_workers=num_workers_eval,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    if prefetch_eval is not None:
        eval_kw["prefetch_factor"] = prefetch_eval

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        drop_last=True, **train_kw,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **eval_kw)
        if val_ds else None
    )
    # Ưu tiên test_ds có nhãn; nếu không có thì dùng test_unlabeled_ds
    # (test_loader lúc này yield (img, filename) thay vì (img, label) —
    # nhớ xử lý riêng ở evaluate/predict, KHÔNG đưa vào tính accuracy/F1).
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **eval_kw)
        if test_ds else
        DataLoader(test_unlabeled_ds, batch_size=batch_size * 2, shuffle=False, **eval_kw)
        if test_unlabeled_ds else None
    )

    _print_summary(
        dcfg, class_names, train_s, val_s or [], test_s, strategy,
        n_test_unlabeled=len(test_unlabeled_paths) if test_unlabeled_paths else 0,
    )
    return train_loader, val_loader, test_loader, class_names