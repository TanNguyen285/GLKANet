import copy
from pathlib import Path
import torch

# Import các hàm từ hệ thống của bạn
try:
    from glkanet.exporter import export_all, load_checkpoint
except ImportError:
    from glkanet.exporter import export_all, load_checkpoint

# =====================================================================
#  CẤU HÌNH ĐƯỜNG DẪN Ở ĐÂY (Thay đổi cho chuẩn khớp với file của bạn)
# =====================================================================
WEIGHTS_PATH = "./runs/exp3/best_f1.pt"       # Đường dẫn file .pt gốc cần xuất
YAML_PATH    = "./glkanet/configs/simple_glka3x3.yaml"     # Đường dẫn file cấu hình .yaml tương ứng
SAVE_DIR     = "./runs/exp3/export_results"             # Thư mục bạn muốn lưu 3 bản sau khi xuất

INPUT_SIZE   = 224                                 # Chiều vuông của ảnh (ví dụ: 224, 256, 384, 512,...)
OPSET_VERSION = 18                                 # Phiên bản ONNX Opset (Mặc định là 18)
# =====================================================================


def main():
    # Chuyển đổi các cấu hình thành dạng Path đối tượng
    weights_p = Path(WEIGHTS_PATH)
    yaml_p = Path(YAML_PATH)
    save_d = Path(SAVE_DIR)

    # Kiểm tra nhanh xem file cấu hình và weights có tồn tại hay không
    if not weights_p.exists():
        print(f"[!] Không tìm thấy file checkpoint tại: {weights_p.absolute()}")
        return
    if not yaml_p.exists():
        print(f"[!] Không tìm thấy file cấu hình YAML tại: {yaml_p.absolute()}")
        return

    print(f"[*] Đang sử dụng Checkpoint: {weights_p.name}")
    print(f"[*] Đang sử dụng Cấu hình:   {yaml_p.name}")
    print(f"[*] Thư mục xuất đầu ra:    {save_d.absolute()}\n")

    # 1. Tải mô hình thông qua hàm load_checkpoint có sẵn của bạn
    print("[*] Đang khởi tạo và nạp state_dict vào mô hình...")
    try:
        model = load_checkpoint(pt_path=weights_p, yaml_path=yaml_p, device="cpu")
    except Exception as e:
        print(f"[!] Lỗi khi load mô hình: {e}")
        print("[!] Hãy chắc chắn rằng cấu trúc trong file .yaml khớp hoàn toàn với file .pt này.")
        return

    # 2. Kiểm tra xem file .pt truyền vào có đúng là bản train chưa reparam không
    ckpt = torch.load(weights_p, map_location="cpu", weights_only=True)
    if ckpt.get("deployed", False):
        print("[!] Cảnh báo: File checkpoint này đã được DEPLOY (đã reparam) trước đó rồi.")
        print("[!] Để xuất ra đầy đủ và chính xác 3 bản, bạn nên truyền vào file .pt gốc ngay sau khi train xong.")
        return

    # 3. Tiến hành chạy hàm xuất 3 bản của bạn
    print("[*] Bắt đầu quá trình trích xuất các phiên bản...")
    exported_paths = export_all(
        model=model,
        save_dir=save_d,
        input_size=INPUT_SIZE,
        yaml_path=yaml_p,
        opset=OPSET_VERSION,
        verbose=True
    )

    print("\n[✓] Quá trình xuất hoàn tất thành công!")
    print(f"    -> Bản Train:  {exported_paths['train'].relative_to(save_d.parent)}")
    print(f"    -> Bản Deploy: {exported_paths['deploy'].relative_to(save_d.parent)}")
    print(f"    -> Bản ONNX:   {exported_paths['onnx'].relative_to(save_d.parent)}")


if __name__ == "__main__":
    main()