# glkanet

Lightweight image classification library — copy thư mục `glkanet/` vào project là dùng được.

---

## Cấu trúc project

```
your_project/
├── glkanet/               ← thư viện (copy vào đây)
├── simple_glka.yaml       ← kiến trúc model (ít thay đổi)
└── configs/
    ├── dataset.yaml       ← đổi dataset → chỉ sửa file này
    └── train.yaml         ← đổi hyperparams → sửa file này
```

---

## Quickstart

### Python API

```python
from glkanet import GLKA

# Train
model = GLKA("simple_glka.yaml")
model.train("configs/train.yaml")

# Override param mà không cần sửa yaml
model.train("configs/train.yaml", epochs=50, device="cuda", lr=0.005)

# Load checkpoint
model = GLKA.from_checkpoint(
    "runs/exp1/weights/best_train.pt",
    "simple_glka.yaml",
)

# Evaluate
model.val("configs/train.yaml", split="val")
model.val("configs/train.yaml", split="test")

# Export 3 bản
model.export()
# → runs/exp1/weights/best_train.pt     (chưa reparam)
# → runs/exp1/weights/best_deploy.pt    (đã reparam)
# → runs/exp1/weights/best_deploy.onnx  (ONNX opset 18)

# Predict
indices, names = model.predict(["img1.jpg", "img2.jpg"])
model.predict(torch.randn(4, 3, 224, 224))  # Tensor cũng được
```

### CLI

```bash
python -m glkanet train  --cfg configs/train.yaml
python -m glkanet train  --cfg configs/train.yaml --epochs 50 --device cuda
python -m glkanet val    --cfg configs/train.yaml --split test
python -m glkanet export --weights runs/exp1/weights/best_train.pt --model simple_glka.yaml
python -m glkanet info   --cfg configs/train.yaml --model simple_glka.yaml
```

---

## configs/dataset.yaml

```yaml
# Đổi dataset → chỉ sửa file này
path:  C:/Users/ThisPC/Desktop/MyDataset

train: train       # relative so với path
val:   val         # bỏ trống → tự split val_ratio% từ train
test:  test        # bỏ trống → tự split test_ratio% từ train

# nc: 5            # optional — tự đếm nếu bỏ
# names: [cat, dog, bird, fish, rabbit]   # optional
```

### Cấu trúc folder dataset

**Standard** — `split/class/img.jpg`:
```
MyDataset/
├── train/
│   ├── cat/   img1.jpg ...
│   └── dog/   img1.jpg ...
├── val/        (optional)
└── test/       (optional)
```

**CCMT-style** — `group/split/class/img.jpg`:
```
MyDataset/
├── Cashew/
│   ├── train_set/  healthy/  diseased/
│   └── test_set/   healthy/  diseased/
└── Tomato/
    ├── train_set/  ...
    └── test_set/   ...
```
Class name sẽ là `Cashew_healthy`, `Tomato_diseased`, ...

### Auto split strategy

| dataset.yaml có | Strategy |
|---|---|
| train + val + test | Giữ nguyên cả 3 |
| train + val | Giữ nguyên, test = None |
| train + test | val = `val_ratio`% từ train (stratified) |
| train only | Tự chia `(1-val-test)` / `val_ratio` / `test_ratio` |

---

## configs/train.yaml

```yaml
model_yaml: simple_glka.yaml       # kiến trúc model
data:        configs/dataset.yaml  # data config

img_size: 224

augment:
  crop_scale: [0.7, 1.0]
  flip_h:     0.5
  flip_v:     0.3
  color_jitter:
    brightness: 0.3
    contrast:   0.3
    saturation: 0.2
    hue:        0.05

train:
  epochs:     100
  batch_size: 64
  seed:       42
  val_ratio:  0.1
  test_ratio: 0.1

  optimizer:
    type: SGD       # SGD | AdamW
    lr:   0.01
    momentum:     0.9
    weight_decay: 0.0005
    nesterov:     true

  scheduler:
    type:    CosineAnnealingLR   # CosineAnnealingLR | StepLR
    eta_min: 0.000001

hardware:
  device:      auto   # auto | cuda | cpu
  num_workers: 4      # Windows lỗi multiprocessing → đổi 0

logging:
  runs_dir:      runs
  tsne_interval: 20   # 0 = tắt t-SNE
  eval_test:     true

export:
  enabled: true
  opset:   18
```

---

## Output sau train

```
runs/exp1/
├── weights/
│   ├── best_f1.pt             ← checkpoint theo F1
│   ├── best_loss.pt           ← checkpoint theo Loss
│   ├── best_train.pt          ← export chưa reparam
│   ├── best_deploy.pt         ← export đã reparam
│   └── best_deploy.onnx       ← ONNX opset 18
├── training_curves.png
├── cm_val_best_f1.png
├── cm_test.png
├── tsne_test.png
├── epoch_reports.txt
├── report_val_best_f1.txt
└── report_test.txt
```

---

## Requirements

```
torch>=2.1
torchvision
pyyaml
numpy
pillow
scikit-learn
matplotlib
seaborn
onnx
onnxscript
```
