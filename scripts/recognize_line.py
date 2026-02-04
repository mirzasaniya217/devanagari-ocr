import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "model/devanagari_model.h5"
INDEX_TO_CLASS_PATH = "model/index_to_class.pkl"

IMG_SIZE = (128, 128)
TOP_K = 3
CONF_THRESHOLD = 0.50   # 50% relative confidence (honest OCR cutoff)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

with open(INDEX_TO_CLASS_PATH, "rb") as f:
    index_to_class = pickle.load(f)


# ---------------- DEVANAGARI MAP ----------------
consonants = [
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
    'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
    'प','फ','ब','भ','म','य','र','ल','व','श',
    'ष','स','ह','क्ष','त्र','ज्ञ'
]

vowels = ['अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ']
numerals = ['०','१','२','३','४','५','६','७','८','९']


def class_to_char(class_name: str) -> str:
    """
    Convert dataset class name → actual Devanagari character
    """
    try:
        if class_name.startswith("consonants_"):
            return consonants[int(class_name.split("_")[1]) - 1]

        if class_name.startswith("vowels_"):
            return vowels[int(class_name.split("_")[1]) - 1]

        if class_name.startswith("numerals_"):
            return numerals[int(class_name.split("_")[1])]

    except Exception:
        pass

    return "?"


# ---------------- PREDICTION ----------------
def predict_character(char_img: np.ndarray, top_k: int = TOP_K):
    """
    Predict one segmented character image.

    Returns:
        final_char (str)
        candidates (list of dicts with percentage confidence)
    """

    # ---------- SAFETY CHECK ----------
    if char_img is None or char_img.size == 0:
        return "?", []

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ---------- MODEL PREDICTION ----------
    preds = model.predict(img, verbose=0)[0]

    top_indices = np.argsort(preds)[-top_k:][::-1]

    # Top-1 & Top-2 normalization (key fix)
    top_prob = preds[top_indices[0]]
    second_prob = preds[top_indices[1]] if len(top_indices) > 1 else 1e-6
    denom = top_prob + second_prob

    candidates = []

    for idx in top_indices:
        class_name = index_to_class.get(idx, "")
        char = class_to_char(class_name)

        # Relative (conditional) confidence
        confidence = preds[idx] / denom

        candidates.append({
            "char": char,
            "confidence": round(confidence * 100, 2)  # percentage
        })

    # ---------- FINAL DECISION ----------
    final_char = candidates[0]["char"]
    final_conf = candidates[0]["confidence"] / 100.0

    if final_conf < CONF_THRESHOLD:
        final_char = "?"

    return final_char, candidates