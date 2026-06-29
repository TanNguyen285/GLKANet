Ý bạn là muốn **một file duy nhất** chứa toàn bộ nội dung của file `README.md` để bạn chỉ cần copy một phát là xong đúng không?

Dưới đây là toàn bộ nội dung file mã nguồn Markdown, bạn chỉ cần bấm nút **Copy** ở góc block code này rồi paste đè vào file `README.md` của mình là chuẩn bài:

```markdown
# ⚡ GLKAnet: Lightweight Image Classification Library

Lightweight image classification library (YOLOv8-style) — copy thư mục `glkanet/` vào project là dùng được.

---

## 📂 Cấu trúc project

```text
your_project/
├── glkanet/               # 📦 Thư viện (copy vào đây)
├── simple_glka.yaml       # 🏗️ Kiến trúc model (YOLO-style)
└── glkanet/configs/       # ⚙️ Thư mục chứa cấu hình
    ├── dataset.yaml       # 🖼️ Cấu hình dataset
    └── train.yaml         # 🚀 Cấu hình siêu tham số (hyperparams)

```

---

## 🚀 Quickstart

### 🐍 Python API (`train.py`)

```python
from glkanet import GLKA

def main():
    # 1. Khởi tạo mô hình
    model = GLKA("simple_glka.yaml")

    print("--- Bắt đầu huấn luyện ---")
    
    # 2. Train mô hình sử dụng cấu hình YAML
    model.train("glkanet/configs/train.yaml")
    
    # Ghi đè tham số nhanh nếu cần (không bắt buộc)
    # model.train("glkanet/configs/train.yaml", epochs=50, device="cuda", lr=0.005)
    
    print("--- Huấn luyện xong! Tự động xuất mô hình ---")
    
    # 3. Xuất ra các phiên bản deploy
    model.export()

    # --- Các hàm bổ trợ khác ---
    # Load checkpoint: model = GLKA.from_checkpoint("runs/exp1/weights/best_train.pt", "simple_glka.yaml")
    # Đánh giá:        model.val("glkanet/configs/train.yaml", split="test")
    # Dự đoán:         indices, names = model.predict(["img1.jpg", "img2.jpg"])
    
if __name__ == "__main__":
    main()

```

### 💻 CLI

```bash
# Train với cấu hình mặc định
python -m glkanet train --cfg glkanet/configs/train.yaml

# Train kèm ghi đè tham số nhanh
python -m glkanet train --cfg glkanet/configs/train.yaml --epochs 50 --device cuda

# Đánh giá mô hình trên tập test
python -m glkanet val   --cfg glkanet/configs/train.yaml --split test

# Export mô hình thủ công từ file weights
python -m glkanet export --weights runs/exp1/weights/best_train.pt --model simple_glka.yaml

# Xem thông tin cấu hình hệ thống
python -m glkanet info   --cfg glkanet/configs/train.yaml --model simple_glka.yaml

```

---

## 📊 Output sau train

Toàn bộ kết quả, đồ thị và file weights sẽ tự động xuất ra trong thư mục `runs/expX/`:

```text
runs/exp1/
├── weights/
│   ├── best_f1.pt            🔥 Checkpoint tốt nhất theo F1-score
│   ├── best_loss.pt          📉 Checkpoint tốt nhất theo Loss
│   ├── best_train.pt         📦 Checkpoint gốc chưa reparam
│   ├── best_deploy.pt        🚀 Checkpoint đã reparam (tối ưu inference)
│   └── best_deploy.onnx      🌐 Mô hình ONNX (Opset 18)
├── training_curves.png       📈 Đồ thị Loss và Accuracy qua từng epoch
├── cm_val_best_f1.png        🧩 Ma trận nhầm lẫn (Confusion Matrix) tập Val
├── cm_test.png               🧩 Ma trận nhầm lẫn tập Test
├── tsne_test.png             🎨 Biểu đồ phân cụm đặc trưng t-SNE
├── epoch_reports.txt         📝 Nhật ký chi tiết của từng epoch
├── report_val_best_f1.txt    📋 Báo cáo chỉ số chi tiết tập Val
└── report_test.txt           📋 Báo cáo chỉ số chi tiết tập Test

```

---

## 🛠️ Requirements

```text
torch >= 2.1
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

```

```