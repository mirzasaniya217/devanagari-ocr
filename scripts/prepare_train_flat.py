import os
import shutil

SRC_DIR = "data/train"
DST_DIR = "data/train_flat"

os.makedirs(DST_DIR, exist_ok=True)

for category in ["consonants", "vowels", "numerals"]:
    cat_path = os.path.join(SRC_DIR, category)

    for cls in os.listdir(cat_path):
        src_cls_path = os.path.join(cat_path, cls)
        if not os.path.isdir(src_cls_path):
            continue

        dst_cls_name = f"{category}_{cls}"
        dst_cls_path = os.path.join(DST_DIR, dst_cls_name)
        os.makedirs(dst_cls_path, exist_ok=True)

        for img in os.listdir(src_cls_path):
            src_img = os.path.join(src_cls_path, img)
            dst_img = os.path.join(dst_cls_path, img)
            shutil.copy(src_img, dst_img)

print(" Dataset flattened into data/train_flat")