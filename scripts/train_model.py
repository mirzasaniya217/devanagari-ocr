import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
DATA_DIR = "data/train_flat"
MODEL_DIR = "model"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- DATA (OCR-SAFE) ----------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=2,        # VERY small
    zoom_range=0.05,         # small
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f" Detected {NUM_CLASSES} classes")

# ---------------- MODEL ----------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Phase 1: freeze backbone
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n Phase 1: Feature extraction")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1
)

# ---------------- PHASE 2: FINE TUNING ----------------
print("\n Phase 2: Fine-tuning top layers")

for layer in base_model.layers[-40:]:   # unfreeze top 40 layers
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),  # LOWER LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2
)

# ---------------- SAVE ----------------
model.save(os.path.join(MODEL_DIR, "devanagari_model.h5"))

index_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open(os.path.join(MODEL_DIR, "index_to_class.pkl"), "wb") as f:
    pickle.dump(index_to_class, f)

print("\n Model and mappings saved successfully")