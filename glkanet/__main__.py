"""glkanet/__main__.py — CLI entry point.

Dùng:
    python -m glkanet train  --cfg configs/ccmt.yaml
    python -m glkanet val    --cfg configs/ccmt.yaml --weights runs/exp1/weights/best_train.pt
    python -m glkanet export --weights runs/exp1/weights/best_train.pt --yaml simple_glka.yaml
    python -m glkanet info   --cfg configs/ccmt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────────────────────

def cmd_train(args):
    from glkanet import GLKA
    model = GLKA(args.model) if args.model else GLKA()
    kwargs = {}
    if args.epochs:  kwargs["epochs"]     = args.epochs
    if args.batch:   kwargs["batch_size"] = args.batch
    if args.device:  kwargs["device"]     = args.device
    if args.lr:      kwargs["lr"]         = args.lr
    model.train(args.cfg, **kwargs)


def cmd_val(args):
    from glkanet import GLKA
    if args.weights:
        model = GLKA.from_checkpoint(args.weights, args.model or _guess_yaml(args.weights))
    else:
        model = GLKA(args.model)
    model.val(args.cfg, split=args.split)


def cmd_export(args):
    from glkanet import GLKA
    model = GLKA.from_checkpoint(args.weights, args.model)
    model.export(
        save_dir   = args.save_dir or Path(args.weights).parent,
        input_size = args.img_size,
        opset      = args.opset,
    )


def cmd_info(args):
    """In thông tin model + data config."""
    import yaml, torch
    from glkanet import GLKA, DataConfig

    print(f"\n{'='*55}")
    print(f"glkanet info")

    if args.model:
        p = Path(args.model)
        if p.exists():
            model = GLKA(p)
            net   = model._build_with_nc(2)   # dummy nc
            total = sum(p.numel() for p in net.parameters())
            print(f"\n[Model yaml]  {args.model}")
            net.info()

    if args.cfg:
        cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
        hw  = cfg.get("hardware", {})
        tr  = cfg.get("train",    {})
        print(f"\n[Train config] {args.cfg}")
        print(f"  data     : {cfg.get('data', 'N/A')}")
        print(f"  epochs   : {tr.get('epochs')}")
        print(f"  batch    : {tr.get('batch_size')}")
        print(f"  optimizer: {tr.get('optimizer', {}).get('type')}  "
              f"lr={tr.get('optimizer', {}).get('lr')}")
        print(f"  scheduler: {tr.get('scheduler', {}).get('type')}")
        print(f"  device   : {hw.get('device', 'auto')}")
        print(f"  workers  : {hw.get('num_workers', 4)}")

        # Data yaml
        data_raw = cfg.get("data")
        if data_raw:
            data_path = Path(data_raw)
            if not data_path.is_absolute():
                data_path = (Path(args.cfg).parent / data_path).resolve()
            if data_path.exists():
                dc = DataConfig(data_path)
                print(f"\n[Data yaml]  {data_path.name}")
                print(f"  train : {dc.train_dir}")
                print(f"  val   : {dc.val_dir   or '(auto-split)'}")
                print(f"  test  : {dc.test_dir  or '(auto-split)'}")
                print(f"  nc    : {dc.nc        or '(auto-scan)'}")

    print(f"\n  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()} "
          f"({'available' if torch.cuda.is_available() else 'not available'})")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _guess_yaml(weights_path: str) -> str:
    """Thử tìm model yaml trong thư mục weights."""
    w = Path(weights_path)
    for candidate in [
        w.parent / "simple_glka.yaml",
        w.parent.parent / "simple_glka.yaml",
        w.parent.parent.parent / "simple_glka.yaml",
        Path("simple_glka.yaml"),
    ]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Không tìm thấy model yaml. Truyền --model path/to/simple_glka.yaml"
    )


# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glkanet",
        description="GLKANet — lightweight image classification",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── train ──
    tr = sub.add_parser("train", help="Train model")
    tr.add_argument("--cfg",    required=True,  help="Train config yaml")
    tr.add_argument("--model",  default=None,   help="Model yaml (override cfg)")
    tr.add_argument("--epochs", type=int,       help="Override epochs")
    tr.add_argument("--batch",  type=int,       help="Override batch size")
    tr.add_argument("--device", default=None,   help="cuda | cpu")
    tr.add_argument("--lr",     type=float,     help="Override learning rate")

    # ── val ──
    vl = sub.add_parser("val", help="Evaluate model")
    vl.add_argument("--cfg",     required=True,        help="Train config yaml")
    vl.add_argument("--weights", default=None,         help="Checkpoint .pt")
    vl.add_argument("--model",   default=None,         help="Model yaml")
    vl.add_argument("--split",   default="val",
                    choices=["val", "test"],            help="Split to evaluate")

    # ── export ──
    ex = sub.add_parser("export", help="Export weights (train/deploy/onnx)")
    ex.add_argument("--weights",  required=True,  help="Checkpoint .pt")
    ex.add_argument("--model",    required=True,  help="Model yaml")
    ex.add_argument("--save-dir", dest="save_dir", default=None,
                    help="Output dir (default: same as weights)")
    ex.add_argument("--img-size", dest="img_size", type=int, default=224)
    ex.add_argument("--opset",    type=int, default=18)

    # ── info ──
    inf = sub.add_parser("info", help="Print model + data config info")
    inf.add_argument("--cfg",   default=None, help="Train config yaml")
    inf.add_argument("--model", default=None, help="Model yaml")

    return p


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    # Thêm thư mục cha vào sys.path để import glkanet khi chạy standalone
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    parser  = build_parser()
    args    = parser.parse_args()

    dispatch = {
        "train":  cmd_train,
        "val":    cmd_val,
        "export": cmd_export,
        "info":   cmd_info,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
