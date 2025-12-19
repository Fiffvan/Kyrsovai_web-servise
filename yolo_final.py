import os
import cv2

SRC_ROOT = "cars_brands_labels"
DST_ROOT = "cars_brands_cropped_txt"
SPLITS = ["train", "val"]

PADDING = 0.01

for split in SPLITS:
    src_split = os.path.join(SRC_ROOT, split)
    dst_split = os.path.join(DST_ROOT, split)
    os.makedirs(dst_split, exist_ok=True)

    for brand in os.listdir(src_split):
        src_brand = os.path.join(src_split, brand)
        dst_brand = os.path.join(dst_split, brand)

        if not os.path.isdir(src_brand):
            continue

        os.makedirs(dst_brand, exist_ok=True)

        for file in os.listdir(src_brand):
            if not file.endswith(".jpg"):
                continue

            img_path = os.path.join(src_brand, file)
            txt_path = img_path.replace(".jpg", ".txt")

            if not os.path.exists(txt_path):
                continue

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            with open(txt_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if len(lines) == 0:
                # нет разметки — пропускаем
                continue

            parts = lines[0].split()
            if len(parts) != 5:
                continue

            _, xc, yc, bw, bh = map(float, parts)

            # YOLO → pixel
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            dx = int((x2 - x1) * PADDING)
            dy = int((y2 - y1) * PADDING)

            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)

            crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(dst_brand, file), crop)

print("✅ Кроп по txt-аннотациям завершён")
