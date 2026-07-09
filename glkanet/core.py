"""glkanet/core.py — Class GLKA: entry point duy nhất.

Dùng như YOLO:
    from glkanet import GLKA

    # Train
    model = GLKA("simple_glka.yaml")
    model.train("configs/ccmt.yaml")

    # Resume train từ checkpoint đã lưu (last.pt / best_f1.pt / best_loss.pt)
    model.train("configs/ccmt.yaml", resume_ckpt="runs/exp1/last.pt")

    # Val / predict
    model.val("configs/ccmt.yaml")

    # Export
    model.export()

    # Load từ checkpoint
    model = GLKA.from_checkpoint("runs/exp1/weights/best_train.pt",
                                  "simple_glka.yaml")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import yaml

# ── Thêm thư mục cha của glkanet vào sys.path để import nội bộ ──
_PKG_DIR = Path(__file__).parent          # glkanet/
_ROOT    = _PKG_DIR.parent               # thư mục chứa glkanet/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ──────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────

def _load_cfg(cfg_path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    hw = cfg.setdefault("hardware", {})
    if hw.get("device", "auto") == "auto":
        hw["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def _resolve_model_yaml(cfg: dict, cfg_path: Path) -> Path:
    rel = cfg.get("model_yaml", "simple_glka.yaml")
    candidate = cfg_path.parent.parent / rel
    if candidate.exists():
        return candidate
    candidate2 = _PKG_DIR.parent / rel
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(
        f"Không tìm thấy model yaml: {rel}\n"
        f"Đặt file yaml ở cùng cấp với thư mục glkanet/"
    )


# ──────────────────────────────────────────────────────────────
# GLKA — main class
# ──────────────────────────────────────────────────────────────

class GLKA:
    def __init__(self, model_yaml: str | Path | None = None):
        self.model_yaml  = Path(model_yaml) if model_yaml else None
        self._model: nn.Module | None = None
        self._class_names: List[str]  = []
        self._cfg: dict               = {}
        self._save_dir: Path | None   = None

        if self.model_yaml is not None:
            if not self.model_yaml.exists():
                raise FileNotFoundError(f"Model yaml không tồn tại: {self.model_yaml}")

    # ── Build / load model ────────────────────────────────────

    def _build_with_nc(self, num_classes: int) -> nn.Module:
        import copy, tempfile, os
        cfg = yaml.safe_load(self.model_yaml.read_text())
        cfg["nc"] = num_classes
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name
        try:
            from glkanet.builder import build_from_yaml
            return build_from_yaml(tmp_path)
        finally:
            os.unlink(tmp_path)

    # ── train ─────────────────────────────────────────────────

    def train(
        self,
        cfg: str | Path,
        resume_ckpt: str | Path | None = None,
        **overrides,
    ) -> dict:
        from glkanet.data import get_data_loaders
        from glkanet.trainer import Trainer, create_save_dir, set_seed

        cfg_path  = Path(cfg)
        train_cfg = _load_cfg(cfg_path, _flatten_overrides(overrides))
        self._cfg = train_cfg

        if self.model_yaml is None:
            self.model_yaml = _resolve_model_yaml(train_cfg, cfg_path)

        set_seed(train_cfg["train"].get("seed", 42))

        data_yaml_path = _resolve_data_yaml(train_cfg, cfg_path)
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
            data_yaml  = data_yaml_path,
            train_cfg  = train_cfg,
            hw_cfg     = train_cfg["hardware"],
            batch_size = train_cfg["train"]["batch_size"],
        )
        self._class_names = class_names

        self._model = self._build_with_nc(len(class_names))
        self._model.info()

        runs_dir = str(_ROOT / train_cfg.get("logging", {}).get("runs_dir", "runs"))

        if resume_ckpt is not None:
            # Resume: dùng lại đúng thư mục cha của checkpoint (vd runs/exp1),
            # KHÔNG tạo exp mới — để log/ckpt tiếp tục nối đúng vào chỗ cũ
            # thay vì bị tách ra làm hai thư mục exp khác nhau.
            resume_ckpt = Path(resume_ckpt)
            if not resume_ckpt.exists():
                raise FileNotFoundError(f"resume_ckpt không tồn tại: {resume_ckpt}")
            self._save_dir = resume_ckpt.resolve().parent
            print(f"[GLKA] Resume từ {resume_ckpt}  save={self._save_dir}")
        else:
            self._save_dir = create_save_dir(runs_dir)
            print(f"[GLKA] device={train_cfg['hardware']['device']}  "
                  f"save={self._save_dir}")

        trainer = Trainer(
            model       = self._model,
            cfg         = train_cfg,
            save_dir    = self._save_dir,
            class_names = class_names,
            resume_ckpt = resume_ckpt,
        )
        result = trainer.run(
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            model_yaml   = self.model_yaml,
            cfg_path     = cfg_path,
            data_yaml    = data_yaml_path,
        )
        self._save_dir = result["save_dir"]
        return result

    # ── val ───────────────────────────────────────────────────

    def val(
        self,
        cfg:     str | Path,
        weights: str | Path | None = None,
        split:   str = "val",
        **overrides,
    ) -> tuple[float, float]:
        from glkanet.data import get_data_loaders
        from glkanet.trainer import evaluate
        from glkanet.logger import save_report, plot_confusion_matrix
        from sklearn.metrics import accuracy_score, f1_score

        cfg_path  = Path(cfg)
        train_cfg = _load_cfg(cfg_path, _flatten_overrides(overrides))
        device    = torch.device(train_cfg["hardware"]["device"])

        data_yaml_path = _resolve_data_yaml(train_cfg, cfg_path)
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
            data_yaml  = data_yaml_path,
            train_cfg  = train_cfg,
            hw_cfg     = train_cfg["hardware"],
            batch_size = train_cfg["train"]["batch_size"],
        )
        self._class_names = class_names

        if weights is not None:
            self.load(weights, cfg_path)
        elif self._model is None:
            raise RuntimeError(
                "Chưa có model. Gọi .train() hoặc .load() trước, "
                "hoặc truyền weights= vào .val()"
            )

        self._model.to(device).eval()
        criterion = nn.CrossEntropyLoss()
        loader    = val_loader if split == "val" else test_loader

        loss, preds, labels, feats = evaluate(
            self._model, loader, criterion, device)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="macro", zero_division=0)

        out_dir = self._save_dir or Path("runs/val")
        out_dir.mkdir(parents=True, exist_ok=True)

        save_report(preds, labels, class_names, out_dir, tag=split)
        plot_confusion_matrix(preds, labels, class_names, out_dir, split)

        print(f"\n[val/{split}]  acc={acc*100:.2f}%  f1={f1:.4f}")
        return acc, f1

    # ── export ────────────────────────────────────────────────

    def export(
        self,
        save_dir:   str | Path | None = None,
        input_size: int = 224,
        opset:      int = 18,
    ) -> dict:
        """Export 3 bản: best_train.pt / best_deploy.pt / best_deploy.onnx
        + tự động copy yaml kiến trúc vào weights/.

        Args:
            save_dir:   thư mục lưu (mặc định = thư mục exp cuối)
            input_size: kích thước ảnh vuông
            opset:      ONNX opset version

        Returns:
            {"train": Path, "deploy": Path, "onnx": Path, "yaml": Path}
        """
        from glkanet.exporter import export_all

        if self._model is None:
            raise RuntimeError("Chưa có model. Gọi .train() hoặc .load() trước.")
        if self.model_yaml is None:
            raise RuntimeError("Không có model_yaml. Khởi tạo GLKA('path/to/yaml') trước.")

        out = Path(save_dir) if save_dir else (self._save_dir or Path("runs/export"))
        out.mkdir(parents=True, exist_ok=True)

        self._model.eval()
        return export_all(
            model      = self._model,
            save_dir   = out,
            yaml_path  = self.model_yaml,
            input_size = input_size,
            opset      = opset,
            verbose    = True,
        )

    # ── predict ───────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images,
        device: str = "cpu",
    ) -> tuple[list, list]:
        if self._model is None:
            raise RuntimeError("Chưa có model. Gọi .train() hoặc .load() trước.")

        dev = torch.device(device)
        self._model.to(dev).eval()

        if not isinstance(images, torch.Tensor):
            images = _load_images(images, img_size=224)

        images    = images.to(dev)
        logits, _ = self._model(images)
        indices   = torch.argmax(logits, dim=1).cpu().tolist()
        names     = (
            [self._class_names[i] for i in indices]
            if self._class_names
            else indices
        )
        return indices, names

    # ── load / save ───────────────────────────────────────────

    def load(
        self,
        pt_path:  str | Path,
        cfg_path: str | Path | None = None,
    ) -> "GLKA":
        pt = Path(pt_path)
        if not pt.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {pt}")

        ckpt   = torch.load(pt, map_location="cpu", weights_only=True)
        device = torch.device(
            self._cfg.get("hardware", {}).get("device", "cpu"))

        if self._model is None:
            if self.model_yaml is None:
                raise RuntimeError(
                    "Cần truyền model_yaml vào GLKA() để load checkpoint.")
            sd = ckpt["state_dict"]
            linear_keys = [k for k in sd if k.startswith("head.classifier.") and k.endswith(".weight") and sd[k].dim() == 2]
            last_key = max(linear_keys, key=lambda k: int(k.split(".")[2]))
            nc = sd[last_key].shape[0]
            self._model = self._build_with_nc(nc)

        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(device).eval()

        if ckpt.get("deployed", False):
            self._model.switch_to_deploy()

        print(f"[GLKA] Loaded {pt.name}  "
              f"(deployed={ckpt.get('deployed', False)})")
        return self

    # ── class method ──────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        pt_path:    str | Path,
        model_yaml: str | Path,
    ) -> "GLKA":
        obj = cls(model_yaml)
        obj.load(pt_path)
        return obj

    # ── info ──────────────────────────────────────────────────

    def info(self) -> None:
        if self._model is None:
            print("[GLKA] Model chưa được khởi tạo.")
            return
        self._model.info()

    def __repr__(self) -> str:
        nc    = len(self._class_names)
        built = self._model is not None
        return (f"GLKA(yaml={self.model_yaml}, "
                f"built={built}, classes={nc})")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _resolve_data_yaml(cfg: dict, cfg_path: Path) -> Path:
    raw = cfg.get("data", None)
    if raw is None:
        raise ValueError(
            "Train config thiếu 'data:'\n"
            "Thêm vào configs/train.yaml:\n"
            "  data: configs/dataset.yaml"
        )
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()
    from_cfg = (cfg_path.parent / p).resolve()
    if from_cfg.exists():
        return from_cfg
    raise FileNotFoundError(
        f"Không tìm thấy data yaml: '{raw}'\n"
        f"  Đã thử: {from_cwd}\n"
        f"  Đã thử: {from_cfg}\n"
        f"Kiểm tra lại 'data:' trong {cfg_path.name}"
    )


def _flatten_overrides(overrides: dict) -> dict:
    mapping = {
        "epochs":     "train.epochs",
        "batch_size": "train.batch_size",
        "lr":         "train.optimizer.lr",
        "device":     "hardware.device",
        "val_ratio":  "data.val_ratio",
        "workers":    "hardware.num_workers",
        "seed":       "train.seed",
    }
    return {mapping.get(k, k): v for k, v in overrides.items()}


def _load_images(paths: list, img_size: int = 224) -> torch.Tensor:
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])