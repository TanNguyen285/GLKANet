import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image

# Gọi thẳng hàm load từ pipeline hệ thống của ông
from glkanet.exporter import load_checkpoint

# =====================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THỰC TẾ TRÊN MÁY ÔNG
# =====================================================================
PATH_TO_MODEL_WEIGHTS = r"C:/Users/ThisPC/Documents/GitHub/Simple_GLKA/runs/export_results/weights/best_train.pt"
DATASET_DIR = r"C:\Users\ThisPC\Desktop\Dataset for Crop Pest and Disease Detection\Dataset for Crop Pest and Disease Detection\Raw Data\CCMT Dataset"
OUTPUT_DIR = "./gradcam_pt_outputs"

NUM_IMAGES_PER_CLASS = 2
IMAGE_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. KHỞI TẠO VÀ NẠP MODEL HOÀN CHỈNH
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang chạy Grad-CAM trên thiết bị: {device}")

model = load_checkpoint(
    pt_path=PATH_TO_MODEL_WEIGHTS,
    yaml_path="glkanet/simple_glka.yaml",
    device=str(device)
)

# Chọn layer tích chập cuối cùng của backbone_layers (Block 6)
target_layers = [model.backbone_layers[6]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================================
# 3. QUÉT ĐỆ QUY TỰ ĐỘNG TÌM TẤT CẢ CÁC LỚP BỆNH THỰC TẾ
# =====================================================================
print("[*] Đang quét toàn bộ các thư mục con để tìm lớp bệnh...")
all_class_dirs = []

# Quét xuyên qua tất cả các tầng thư mục con
for root, dirs, files in os.walk(DATASET_DIR):
    # Nếu thư mục chứa file ảnh trực tiếp thì đây chính là một lớp bệnh
    has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
    if has_images:
        all_class_dirs.append(root)

all_class_dirs = sorted(all_class_dirs)
print(f"[*] Tìm thấy chính xác {len(all_class_dirs)} lớp bệnh thực tế. Tiến hành xuất ảnh...")

# =====================================================================
# 4. CHẠY GRAD-CAM TRÊN TỪNG LỚP THỰC TẾ
# =====================================================================
for class_idx, class_path in enumerate(all_class_dirs):
    # Lấy tên lớp bệnh thực tế (tên thư mục cuối cùng)
    class_name = os.path.basename(class_path)
    
    print(f" -> Đang xử lý lớp [{class_idx+1}/{len(all_class_dirs)}]: {class_name}")
    
    # Tạo thư mục lưu ảnh đầu ra cho lớp này
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Lấy danh sách ảnh trong thư mục này
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = image_files[:NUM_IMAGES_PER_CLASS]
    
    for img_idx, img_name in enumerate(selected_images):
        img_path = os.path.join(class_path, img_name)
        
        try:
            # Đọc ảnh gốc bằng PIL
            pil_img = Image.open(img_path).convert('RGB')
            
            # Tạo ảnh nền float định dạng chuẩn RGB cho OpenCV trộn ma trận màu
            rgb_img_float = np.array(pil_img, dtype=np.float32) / 255.0
            rgb_img_float = cv2.resize(rgb_img_float, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Đẩy ảnh qua pipeline tensor
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # Chỉ định class mục tiêu cần giải thích đạo hàm
            # Nếu model của ông map nhãn dựa trên alphabet của 22 class, dùng class_idx là chuẩn bài
            targets = [ClassifierOutputTarget(class_idx)]
            
            # Tính toán bản đồ nhiệt toán học (Grad-CAM)
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            # Trộn bản đồ nhiệt dải màu JET lên ảnh thực tế
            cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            # Lưu ảnh kết quả vào thư mục tương ứng
            output_name = f"{class_name}_sample_{img_idx+1}_cam.png"
            cv2.imwrite(os.path.join(class_output_dir, output_name), cam_image)
            
        except Exception as e:
            print(f"    [LỖI] Không thể xử lý ảnh {img_name}: {e}")

print(f"\n[+] HOÀN THÀNH XUẤT XƯỞNG! Ông mở thư mục này ra check ảnh nhé: {os.path.abspath(OUTPUT_DIR)}")