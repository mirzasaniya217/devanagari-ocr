import cv2
import os
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold for poor-quality images
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return img, clean


def detect_lines(original, binary, output_dir="output_lines"):
    os.makedirs(output_dir, exist_ok=True)

    # Horizontal projection
    horizontal_sum = np.sum(binary, axis=1)

    lines = []
    in_line = False
    start = 0

    for i, value in enumerate(horizontal_sum):
        if value > 0 and not in_line:
            start = i
            in_line = True
        elif value == 0 and in_line:
            end = i
            in_line = False
            if end - start > 10:
                lines.append((start, end))

    print(f"Detected {len(lines)} lines")

    for idx, (y1, y2) in enumerate(lines):
        line_img = original[y1:y2, :]
        cv2.imwrite(f"{output_dir}/line_{idx+1}.png", line_img)

    return lines


if __name__ == "__main__":
    image_path = "data/samples/sample.png"

    original, binary = preprocess_image(image_path)
    detect_lines(original, binary)