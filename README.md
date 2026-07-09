
```markdown
# Simple_GLKA

Repo này dùng để train, đánh giá và export mô hình phân loại ảnh theo kiến trúc GLKA. Mục tiêu là chạy được từ đầu đến cuối với cấu hình YAML, sau đó xuất ra file deploy và TFLite.

## 1. Cài đặt ban đầu

Cài các package cần thiết trước khi chạy:

```bash
pip install torch torchvision pyyaml numpy pillow scikit-learn matplotlib seaborn onnx onnxscript
```

Nếu cần export sang TFLite, cài thêm bộ thư viện trong thư mục TFlite:

```bash
pip install -r TFlite/requirements-tflite.txt
```

## 2. Chuẩn bị dữ liệu

Đặt dataset theo cấu trúc thư mục chuẩn hoặc dùng cấu trúc CCMT-style. File cấu hình dữ liệu nằm ở:

- glkanet/configs/dataset.yaml

Ví dụ ngắn:

```yaml
path: C:/path/to/your/dataset
train: train_set
test: test_set
```

## 3. Chỉnh cấu hình train

File cấu hình chính nằm ở:

- glkanet/configs/train.yaml

Bạn có thể chỉnh các mục chính như:

- epochs, batch_size, learning rate
- device: cuda hoặc cpu
- đường dẫn dataset
- export: bật/tắt export ONNX/TFLite

Lưu ý trên Windows nếu gặp lỗi multiprocessing thì đổi `num_workers` xuống `0`.

## 4. Chạy train

Từ thư mục gốc của repo, có thể chạy nhanh như sau:

```bash
python -m glkanet train --cfg glkanet/configs/train.yaml
```

Hoặc dùng script Python sẵn:

```bash
python GLKAnet.py
```

Ví dụ dùng Python API:

```python
from glkanet import GLKA

model = GLKA("glkanet/configs/shuffle_glkav2.yaml")
model.train("glkanet/configs/train.yaml")
model.export()
```

## 5. Kiểm tra / đánh giá

```bash
python -m glkanet val --cfg glkanet/configs/train.yaml --split test
```

## 6. Export sang ONNX / TFLite

Sau khi train xong, repo sẽ tự động export các file ở thư mục `runs/expX/weights/`.

Export thủ công:

```bash
python -m glkanet export --weights runs/exp1/weights/best_train.pt --model glkanet/configs/shuffle_glkav2.yaml
```

Đối với TFLite, repo đã có script riêng ở thư mục TFlite. Trên Windows có thể dùng:

```bash
TFlite\setup_and_run.bat --onnx runs\exp1\weights\best_deploy.onnx --out runs\exp1\weights\tflite --input-size 224 --mode all
```

## 7. Kết quả sau khi chạy

Kết quả sẽ được lưu trong thư mục `runs/expX/` gồm:

- weights: checkpoint và file deploy
- báo cáo huấn luyện và confusion matrix
- đồ thị loss/accuracy

Ví dụ cấu trúc cơ bản:

```text
runs/exp1/
├── weights/
├── epoch_reports.txt
├── report_val_best_f1.txt
├── report_test.txt
└── training_curves.png
```

## 8. Cấu trúc thư mục chính

```text
Simple_GLKA/
├── glkanet/              # code chính: model, trainer, exporter
├── glkanet/configs/      # file YAML cho model/data/train
├── runs/                 # output training và evaluation
├── TFlite/               # script export sang TFLite
├── GLKAnet.py            # entrypoint train nhanh
└── README.md             # hướng dẫn dùng repo
```
Nếu bạn muốn dùng repo này cho dataset riêng, chỉ cần sửa 2 file config: dataset.yaml và train.yaml là đã có thể chạy được.

```