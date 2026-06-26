from glkanet import GLKA
import torch
# Train
model = GLKA("simple_glka.yaml")
model.train("configs/ccmt.yaml")

# Override không cần sửa yaml
model.train("configs/ccmt.yaml", epochs=50, device="cuda", lr=0.005)

# Load checkpoint
model = GLKA.from_checkpoint("runs/exp1/weights/best_train.pt", "simple_glka.yaml")

# Evaluate
model.val("configs/ccmt.yaml", split="test")

# Export 3 bản
model.export()

# Predict
indices, names = model.predict(["img1.jpg", "img2.jpg"])
model.predict(torch.randn(4, 3, 224, 224))  # Tensor cũng được