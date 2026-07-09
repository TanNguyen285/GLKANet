from __future__ import annotations
import copy
import csv
import json
import os
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models as tvm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[warn] Chưa cài timm — bỏ qua nhóm model 2024-2026. pip install timm")

try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] Chưa cài scikit-learn — macro-F1 sẽ được tính tay (chậm hơn).")

from glkanet import GLKA

NUM_CLASSES = 22

# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════
TRAIN_DIR = r"C:\Users\ThisPC\Desktop\Raw Data\CCMT Dataset\train_set"
TEST_DIR  = r"C:\Users\ThisPC\Desktop\Raw Data\CCMT Dataset\test_set"

IMG_SIZE   = 224
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

CROP_SCALE       = (0.7, 1.0)
FLIP_H_P         = 0.5
FLIP_V_P         = 0.3
COLOR_JITTER     = dict(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

EPOCHS       = 10
BATCH_SIZE   = 64
SEED         = 42
VAL_RATIO    = 0.1
HANDLE_IMBALANCE = True
LABEL_SMOOTHING  = 0.0

LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
NESTEROV     = True
ETA_MIN      = 0.000001

DEVICE       = "auto"     # "auto" | "cuda" | "cpu"
NUM_WORKERS  = 8
PIN_MEMORY   = True

RUNS_DIR     = Path("runs_compare")

# GLKANet: đã train xong — chỉ load checkpoint + arch yaml qua glkanet
GLKANET_ARCH_YAML  = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\glkanet\configs\Hybird.yaml"
GLKANET_CHECKPOINT = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\hybird\weights\best_deploy.pt"


def build_shufflenet_v2(num_classes):
    m = tvm.shufflenet_v2_x0_5(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_mobilenet_v3_small(num_classes):
    m = tvm.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m

def _build_timm(name, num_classes):
    return timm.create_model(name, pretrained=False, num_classes=num_classes)

MODEL_REGISTRY = {
    "ShuffleNetV2_x0.5":        build_shufflenet_v2,
    "MobileNetV3_small":        build_mobilenet_v3_small,
}
if HAS_TIMM:
    MODEL_REGISTRY.update({
        "MobileNetV4_conv_s050": lambda nc: _build_timm("mobilenetv4_conv_small_050", nc),
        "RepViT_m0.9":           lambda nc: _build_timm("repvit_m0_9", nc),
        "FastViT_t8":            lambda nc: _build_timm("fastvit_t8", nc),
        "EdgeNeXt_xx_small":     lambda nc: _build_timm("edgenext_xx_small", nc),
        "StarNet_s1":            lambda nc: _build_timm("starnet_s1", nc),
    })

QUICK_TEST_EPOCHS = None   # vd đặt = 2 để test cơ chế; None = dùng EPOCHS thật


# ══════════════════════════════════════════════════════════════════════
# SEED / REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seeded_worker_init_fn(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
def resolve_device(pref):
    pref = (pref or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=CROP_SCALE),
        transforms.RandomHorizontalFlip(p=FLIP_H_P),
        transforms.RandomVerticalFlip(p=FLIP_V_P),
        transforms.ColorJitter(**COLOR_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train_tf, eval_tf


def stratified_val_split(dataset, val_ratio, seed):
    rng = random.Random(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    val_idx, train_idx = [], []
    for label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def make_weighted_sampler(dataset, indices):
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders():
    train_tf, eval_tf = build_transforms()

    full_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    label_ds = datasets.ImageFolder(TRAIN_DIR)
    train_idx, val_idx = stratified_val_split(label_ds, VAL_RATIO, SEED)

    full_val_ds = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)

    train_subset = Subset(full_train_ds, train_idx)
    val_subset   = Subset(full_val_ds, val_idx)
    test_ds      = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    num_classes = len(label_ds.classes)
    assert num_classes == NUM_CLASSES, (
        f"[warn] Số lớp thực tế trong dataset ({num_classes}) khác NUM_CLASSES "
        f"đã set trong train_compare.py ({NUM_CLASSES}) — kiểm tra lại config."
    )

    gen = make_generator(SEED)

    if HANDLE_IMBALANCE:
        sampler = make_weighted_sampler(label_ds, train_idx)
        sampler.generator = make_generator(SEED)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
                                   worker_init_fn=seeded_worker_init_fn, generator=gen)
    else:
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
                                   worker_init_fn=seeded_worker_init_fn, generator=gen)

    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             worker_init_fn=seeded_worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              worker_init_fn=seeded_worker_init_fn)

    print(f"[data] train={len(train_subset)}  val={len(val_subset)}  test={len(test_ds)}  "
          f"num_classes={num_classes}")
    return train_loader, val_loader, test_loader, num_classes


# ══════════════════════════════════════════════════════════════════════
# TRAIN + EVAL
# ══════════════════════════════════════════════════════════════════════
def train_one_model(name, model, train_loader, val_loader, device, epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        train_acc = correct / max(1, total)
        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"[{name}] epoch {epoch+1}/{epochs}  loss={running_loss/max(1,total):.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(1, total)


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Trả về accuracy + macro-F1 trên test set. Yêu cầu model.forward(x) ->
    logits (Tensor) — nếu model gốc trả tuple (logits, feats), phải bọc lại
    trước khi truyền vào đây (xem GLKANetLogitsOnly)."""
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = correct / max(1, total)
    if HAS_SKLEARN:
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
    else:
        macro_f1 = _macro_f1_manual(all_labels, all_preds)
    return acc, macro_f1


def _macro_f1_manual(labels, preds):
    classes = set(labels) | set(preds)
    f1s = []
    for c in classes:
        tp = sum(1 for l, p in zip(labels, preds) if l == c and p == c)
        fp = sum(1 for l, p in zip(labels, preds) if l != c and p == c)
        fn = sum(1 for l, p in zip(labels, preds) if l == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return statistics.mean(f1s) if f1s else 0.0


# ══════════════════════════════════════════════════════════════════════
# GLKANet — load qua thư viện glkanet chính chủ (chỉ eval acc/f1, không train)
# ══════════════════════════════════════════════════════════════════════
class GLKANetLogitsOnly(nn.Module):
    """Bọc model gốc của glkanet để forward() chỉ trả logits."""
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def load_glkanet_deployed(checkpoint_path, arch_yaml, device):
    wrapper = GLKA(arch_yaml)
    wrapper.load(checkpoint_path)
    model = GLKANetLogitsOnly(wrapper._model).to(device).eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# MAIN — chỉ train + đo acc/f1 + lưu checkpoint. KHÔNG đo FPS ở đây.
# Đo FPS riêng bằng bench_fps.py để tránh nhiễu do máy vừa train xong.
# ══════════════════════════════════════════════════════════════════════
def main():
    device = resolve_device(DEVICE)
    print(f"[info] Device: {device}")
    RUNS_DIR.mkdir(exist_ok=True)

    set_seed(SEED)
    train_loader, val_loader, test_loader, num_classes = build_dataloaders()
    epochs = QUICK_TEST_EPOCHS or EPOCHS

    results = []

    # ── GLKANet: đã train xong, chỉ load + eval acc/f1 ──
    print(f"\n{'='*100}\n[MODEL] GLKANet (đã train sẵn — load qua thư viện glkanet)\n{'='*100}")
    try:
        set_seed(SEED)
        glka_model = load_glkanet_deployed(GLKANET_CHECKPOINT, GLKANET_ARCH_YAML, device)
        n_params = sum(p.numel() for p in glka_model.parameters())
        test_acc, macro_f1 = evaluate_full(glka_model, test_loader, device)
        row = {
            "model": "GLKANet", "params_M": round(n_params / 1e6, 4),
            "val_acc": None, "test_acc": round(test_acc, 4), "macro_f1": round(macro_f1, 4),
            "checkpoint": str(GLKANET_CHECKPOINT),
        }
        results.append(row)
        print(f"[GLKANet] params={row['params_M']}M  test_acc={row['test_acc']}  macro_f1={row['macro_f1']}")
    except Exception as ex:
        print(f"[warn] Load/eval GLKANet lỗi: {ex}  "
              f"(kiểm tra lại GLKANET_ARCH_YAML / GLKANET_CHECKPOINT)")

    # ── Các model còn lại: TRAIN từ đầu, reset seed trước mỗi model ──
    for name, builder in MODEL_REGISTRY.items():
        print(f"\n{'='*100}\n[MODEL] {name}\n{'='*100}")
        set_seed(SEED)
        try:
            model = builder(num_classes)
        except Exception as ex:
            print(f"[warn] Không build được model '{name}': {ex}")
            continue

        n_params = sum(p.numel() for p in model.parameters())

        try:
            model, best_val_acc = train_one_model(name, model, train_loader, val_loader, device, epochs)
        except Exception as ex:
            print(f"[warn] Train '{name}' lỗi: {ex}")
            continue

        set_seed(SEED)
        test_acc, macro_f1 = evaluate_full(model, test_loader, device)

        ckpt_path = RUNS_DIR / f"{name}.pt"
        torch.save({"model": model.state_dict(), "num_classes": num_classes}, ckpt_path)

        row = {
            "model": name,
            "params_M": round(n_params / 1e6, 4),
            "val_acc": round(best_val_acc, 4),
            "test_acc": round(test_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "checkpoint": str(ckpt_path),
        }
        results.append(row)
        print(f"[{name}] params={row['params_M']}M  test_acc={row['test_acc']}  macro_f1={row['macro_f1']}")

    # ── In bảng tổng hợp + lưu CSV/JSON (chưa có FPS — chạy bench_fps.py sau) ──
    print(f"\n{'='*100}\nBẢNG TỔNG HỢP (chưa có FPS — chạy bench_fps.py để đo)\n{'='*100}")
    header = f"{'Model':<24}{'Params(M)':>11}{'TestAcc':>10}{'MacroF1':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<24}{r['params_M']:>11}{r['test_acc']:>10}{r['macro_f1']:>10}")

    csv_path = RUNS_DIR / "results_train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    json_path = RUNS_DIR / "results_train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[info] Đã lưu: {csv_path}  và  {json_path}")
    print(f"[info] Chạy `python bench_fps.py` để đo FPS riêng trên máy sạch.")


if __name__ == "__main__":
    main()
