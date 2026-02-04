import os
import cv2
import shutil

from scripts import line_detection
from scripts import char_segmentation
from scripts import recognize_line

CHAR_OUTPUT_DIR = "output_chars"


def run_ocr(image_path: str) -> dict:
    """
    Full OCR pipeline for a single uploaded image.
    """

    os.makedirs(CHAR_OUTPUT_DIR, exist_ok=True)

    # ---------- LINE DETECTION ----------
    original, binary = line_detection.preprocess_image(image_path)
    lines = line_detection.detect_lines(original, binary)

    if not lines:
        lines = [(0, original.shape[0])]

    recognized_lines = []
    all_characters = []
    global_position = 1

    # ---------- PROCESS EACH LINE ----------
    for (y1, y2) in lines:
        line_img = original[y1:y2, :]

        binary_line = char_segmentation.preprocess_line(line_img)

        # RETURNS: [(x, y, w, h), ...]
        boxes = char_segmentation.segment_characters(
            line_img,
            binary_line,
            output_dir=CHAR_OUTPUT_DIR
        )

        line_text = ""

        # ---------- CHARACTER RECOGNITION ----------
        for (x, y, w, h) in boxes:
            char_img = line_img[y:y+h, x:x+w]

            if char_img.size == 0:
                continue

            char_img = cv2.resize(char_img, (128, 128))

            char, candidates = recognize_line.predict_character(char_img)

            line_text += char

            all_characters.append({
                "position": global_position,
                "predicted": char,
                "candidates": candidates
            })

            global_position += 1

        if line_text.strip():
            recognized_lines.append(line_text)

    # ---------- CLEANUP ----------
    try:
        shutil.rmtree(CHAR_OUTPUT_DIR)
    except Exception:
        pass

    return {
        "recognized_text": "\n".join(recognized_lines),
        "characters": all_characters
    }