import cv2
import os
import numpy as np


def preprocess_line(line_img):
    """
    Preprocess a single text line image for character segmentation.
    Designed to be robust to low-quality scans.
    """
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (works for poor lighting / contrast)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Light noise removal (do NOT aggressively remove strokes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return clean


def segment_characters(line_img, binary_img, output_dir=None):
    """
    Segment characters from a single Devanagari text line using
    vertical projection analysis.

    Returns:
        List of bounding boxes: [(x, y, w, h), ...]
    """
    h, w = binary_img.shape

    # Vertical projection
    vertical_sum = np.sum(binary_img, axis=0)

    # Normalize safely
    max_val = np.max(vertical_sum)
    if max_val == 0:
        return []

    vertical_sum = vertical_sum / max_val

    char_regions = []
    in_char = False
    start_x = 0

    for x, val in enumerate(vertical_sum):
        if val > 0.05 and not in_char:
            start_x = x
            in_char = True
        elif val <= 0.05 and in_char:
            end_x = x
            in_char = False

            # Minimum width filter
            if end_x - start_x > 8:
                char_regions.append((start_x, end_x))

    boxes = []

    for (x1, x2) in char_regions:
        char_img = line_img[:, x1:x2]

        # Crop vertical whitespace
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(255 - gray)

        if coords is None:
            continue

        x, y, w_box, h_box = cv2.boundingRect(coords)

        # Final bounding box in line image coordinates
        boxes.append((x1 + x, y, w_box, h_box))

    # Sort left → right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Optional debug save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for idx, (x, y, w_box, h_box) in enumerate(boxes):
            char_img = line_img[y:y+h_box, x:x+w_box]
            char_img = cv2.resize(char_img, (128, 128))
            cv2.imwrite(f"{output_dir}/char_{idx+1}.png", char_img)

    return boxes


# ---------------- DEBUG MODE ----------------
if __name__ == "__main__":
    line_image_path = "output_lines/line_1.png"

    line_img = cv2.imread(line_image_path)
    if line_img is None:
        raise FileNotFoundError("Line image not found")

    binary = preprocess_line(line_img)

    # Debug output (optional)
    cv2.imwrite("debug_binary.png", binary)

    boxes = segment_characters(line_img, binary, output_dir="output_chars")
    print(f"Detected {len(boxes)} characters")