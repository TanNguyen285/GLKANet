"""
make_proxy_subset.py
Tạo subset stratified (theo class) từ CCMT Dataset (bản RAW) để dùng cho proxy training.
Dùng symlink thay vì copy để tiết kiệm dung lượng đĩa và thời gian.

Cách chạy:
    python make_proxy_subset.py --ratio 0.15 --seed 42

Kết quả:
    Tạo thư mục "CCMT Dataset Proxy" cùng cấp với "CCMT Dataset" gốc, cấu trúc:
        CCMT Dataset Proxy/
            train_set/<class_name>/*.jpg (symlink)
            test_set/<class_name>/*.jpg  (symlink, giữ nguyên 100% vì test_set nhỏ, dùng để so proxy vs full)
"""

import argparse
import random
from pathlib import Path

# ============ CHỈNH ĐƯỜNG DẪN Ở ĐÂY NẾU CẦN ============
SOURCE_ROOT = Path(r"C:\Users\ThisPC\Desktop\Raw Data\CCMT Dataset")
TARGET_ROOT = Path(r"C:\Users\ThisPC\Desktop\Raw Data\CCMT Dataset Proxy")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# =========================================================


def list_images(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


def make_subset(split_name: str, ratio: float, seed: int, min_per_class: int = 20):
    """
    split_name: "train_set" hoặc "test_set"
    ratio: tỷ lệ ảnh giữ lại mỗi class (vd 0.15 = 15%)
    min_per_class: số ảnh tối thiểu giữ lại mỗi class dù ratio*count < min_per_class
                   (tránh class hiếm bị còn quá ít ảnh, làm proxy không ổn định)
    """
    src_split = SOURCE_ROOT / split_name
    dst_split = TARGET_ROOT / split_name
    dst_split.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)  # seed riêng theo split để tái lập ổn định

    class_dirs = sorted([d for d in src_split.iterdir() if d.is_dir()])
    print(f"\n[{split_name}] Tìm thấy {len(class_dirs)} class")

    summary = []
    for class_dir in class_dirs:
        images = list_images(class_dir)
        n_total = len(images)
        n_keep = max(min_per_class, int(n_total * ratio))
        n_keep = min(n_keep, n_total)  # không vượt quá số ảnh có sẵn

        rng.shuffle(images)
        chosen = images[:n_keep]

        dst_class_dir = dst_split / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            link_path = dst_class_dir / img_path.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(img_path.resolve())
                except OSError:
                    # Windows cần quyền admin hoặc Developer Mode để symlink.
                    # Fallback: copy nếu symlink thất bại.
                    import shutil
                    shutil.copy2(img_path, link_path)

        summary.append((class_dir.name, n_total, n_keep))
        print(f"  {class_dir.name:30s}  {n_total:5d} -> {n_keep:5d}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.15,
                         help="Tỷ lệ ảnh train giữ lại mỗi class (default 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_full", action="store_true", default=True,
                         help="Giữ 100% test_set (mặc định True, để so proxy vs full accuracy công bằng)")
    args = parser.parse_args()

    print(f"Source: {SOURCE_ROOT}")
    print(f"Target: {TARGET_ROOT}")
    print(f"Train ratio: {args.ratio} | Seed: {args.seed}")

    # Train set: lấy subset theo ratio
    train_summary = make_subset("train_set", ratio=args.ratio, seed=args.seed, min_per_class=20)

    # Test set: giữ nguyên 100% (ratio=1.0) để đánh giá công bằng, so sánh trực tiếp
    # với accuracy của model full-train trên cùng test_set
    test_summary = make_subset("test_set", ratio=1.0, seed=args.seed, min_per_class=1)

    total_train = sum(n for _, _, n in train_summary)
    total_test = sum(n for _, _, n in test_summary)
    print(f"\n=== TỔNG KẾT ===")
    print(f"Train proxy: {total_train} ảnh (từ {sum(n for _,n,_ in train_summary)} ảnh gốc)")
    print(f"Test (giữ nguyên): {total_test} ảnh")
    print(f"\nĐã tạo xong tại: {TARGET_ROOT}")
    print("Nhớ kiểm tra lại vài class có min_per_class=20 xem có bị lấy quá tỷ lệ ratio không "
          "(class ít ảnh gốc sẽ có % giữ lại cao hơn các class khác).")


if __name__ == "__main__":
    main()
