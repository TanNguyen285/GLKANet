
```markdown
# Simple_GLKA

Simple_GLKA is a lightweight PyTorch-based image classification pipeline for training, evaluating, and exporting GLKA-style models from YAML configuration files.

## Overview

This repository is designed for fast experimentation with image classification models. You can:
- define the model architecture in YAML
- point the pipeline to your dataset
- train and evaluate the model
- export checkpoints, ONNX, and TFLite artifacts

## Features

- YAML-driven model definition
- Support for ImageFolder-style datasets and CCMT-style datasets
- Train / validation / test workflow
- Automatic checkpoint saving for best F1 and best loss
- ONNX export and TFLite export support
- Logging, confusion matrices, and training curve plots

## Project Structure

```text
Simple_GLKA/
├── GLKAnet.py                # quick training entrypoint
├── glkanet/                  # core library
│   ├── core.py               # GLKA wrapper API
│   ├── builder.py            # build model from YAML
│   ├── trainer.py            # training loop and evaluation
│   ├── exporter.py           # export logic
│   ├── data/                 # data loaders and dataset parsing
│   └── configs/              # model/data/train YAML files
├── TFlite/                   # standalone TFLite export pipeline
├── scripts/                  # helper scripts
└── runs/                     # training outputs
```

## Installation

Recommended Python version: 3.10+

Install the main dependencies:

```bash
pip install torch torchvision pyyaml numpy pillow scikit-learn matplotlib seaborn onnx onnxscript
```

If you want to export TFLite artifacts, install the additional requirements:

```bash
pip install -r TFlite/requirements-tflite.txt
```

## Quick Start

### 1) Train from the provided script

```bash
python GLKAnet.py
```

This uses the model config from [glkanet/configs/Hybird.yaml](glkanet/configs/Hybird.yaml) and the training config from [glkanet/configs/train.yaml](glkanet/configs/train.yaml).

### 2) Train using Python API

```python
from glkanet import GLKA

model = GLKA("glkanet/configs/Hybird.yaml")
model.train("glkanet/configs/train.yaml")
```

### 3) Train using CLI

```bash
python -m glkanet train --cfg glkanet/configs/train.yaml --model glkanet/configs/Hybird.yaml
```

## Dataset Preparation

The loader supports two common dataset layouts:

### ImageFolder-style

```text
dataset/
├── train/
│   ├── cat/
│   └── dog/
├── val/
└── test/
```

### CCMT-style

```text
dataset/
├── Cashew/
│   ├── train_set/
│   └── test_set/
└── Tomato/
    ├── train_set/
    └── test_set/
```

Update [glkanet/configs/dataset.yaml](glkanet/configs/dataset.yaml) to match your dataset path and split names.

## Configuration Files

- [glkanet/configs/Hybird.yaml](glkanet/configs/Hybird.yaml): model architecture
- [glkanet/configs/dataset.yaml](glkanet/configs/dataset.yaml): dataset path and split configuration
- [glkanet/configs/train.yaml](glkanet/configs/train.yaml): training hyperparameters, device, and export options

## Evaluation

```bash
python -m glkanet val --cfg glkanet/configs/train.yaml --model glkanet/configs/Hybird.yaml --split test --weights runs/exp1/weights/best_f1.pt
```

## Export

### Export ONNX

```bash
python -m glkanet export --weights runs/exp1/weights/best_train.pt --model glkanet/configs/Hybird.yaml
```

### Export TFLite from a separate script

For a standalone TFLite export workflow, you can create your own file and call the exporter inside [TFlite](TFlite). A ready-to-edit template is available at [scripts/export_tflite_template.py](scripts/export_tflite_template.py).

Example:

```bash
python scripts/export_tflite_template.py --onnx runs/exp1/weights/best_deploy.onnx --out runs/exp1/weights/tflite --input-size 224 --mode all
```

## Outputs

After training, results are written to a folder under [runs](runs), for example:

```text
runs/exp1/
├── weights/
│   ├── best_f1.pt
│   ├── best_loss.pt
│   ├── best_train.pt
│   ├── best_deploy.pt
│   └── best_deploy.onnx
├── epoch_reports.txt
├── report_val_best_f1.txt
├── report_test.txt
└── training_curves.png
```

## Notes

- On Windows, if you encounter multiprocessing issues, reduce `num_workers` to `0` in [glkanet/configs/train.yaml](glkanet/configs/train.yaml).
- For GPU training, keep `device: cuda`.
- To resume training from a previous checkpoint:

```python
from glkanet import GLKA

model = GLKA("glkanet/configs/Hybird.yaml")
model.train("glkanet/configs/train.yaml", resume_ckpt="runs/exp1/last.pt")
```
```