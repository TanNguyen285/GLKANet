# scan_broken_images.py
import os
from PIL import Image
from pathlib import Path

ROOT = r"C:\Users\ThisPC\Desktop\CCMT Dataset-Augmented"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
bad = []

for p in Path(ROOT).rglob("*"):
    if p.suffix.lower() not in exts:
        continue
    try:
        with Image.open(p) as im:
            im.load()  # force full decode, không chỉ đọc header
    except Exception as e:
        bad.append(str(p))
        print(f"[BAD] {p} -> {e}")

print(f"\nTổng ảnh hỏng: {len(bad)}")
with open("bad_images.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(bad))

# Xoá luôn nếu muốn:
# for p in bad:
#     os.remove(p)