from glkanet import GLKA

def main():
    # 1. Khởi tạo mô hình dựa trên file kiến trúc ở thư mục gốc
    model = GLKA("glkanet/simple_glka.yaml")

    print("--- Bắt đầu huấn luyện theo cấu hình từ YAML ---")
    
    # 2. Sửa lại đường dẫn: trỏ thẳng vào trong thư mục glkanet/configs/
    model.train("glkanet/configs/train.yaml")
    
    print("--- Huấn luyện xong! Tự động export mô hình ---")
    model.export()

if __name__ == "__main__":
    main()