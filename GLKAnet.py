from glkanet import GLKA

def main():
    model = GLKA("glkanet/configs/dualattention_glkaV1.yaml")

    print("--- Bắt đầu huấn luyện theo cấu hình từ YAML ---")
    model.train("glkanet/configs/train.yaml")
    print("--- Huấn luyện xong! Mô hình đã được export tự động trong quá trình train ---")

if __name__ == "__main__":
    main()