import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("model/devanagari_model.h5")

IMG_SIZE = (128, 128)

def predict(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return class_idx, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    cls, conf = predict(sys.argv[1])
    print("Predicted class index:", cls)
    print("Confidence:", round(conf, 3))