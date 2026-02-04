import os
import shutil

BASE_DIR = "/Users/mirzasaniya/Documents/devanagari_character_recognition"
SRC_DIR = os.path.join(BASE_DIR, "data/train")
DST_DIR = os.path.join(BASE_DIR, "data/train_flat")

os.makedirs(DST_DIR, exist_ok=True)

categories = ["consonants", "vowels", "numerals"]

for category in categories:
    category_path = os.path.join(SRC_DIR, category)

    for class_id in os.listdir(category_path):
        src_class_dir = os.path.join(category_path, class_id)

        if not os.path.isdir(src_class_dir):
            continue

        # Create flattened class folder
        class_name = f"{category}_{class_id}"
        dst_class_dir = os.path.join(DST_DIR, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        for img in os.listdir(src_class_dir):
            if img.lower().endswith((".png", ".jpg", ".jpeg")):
                shutil.copy(
                    os.path.join(src_class_dir, img),
                    os.path.join(dst_class_dir, img)
                )

print("Dataset flattened successfully into data/train_flat")