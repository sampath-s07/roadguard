"""
train_india.py
Trains a CNN on the Indian Traffic Sign Dataset.
Run: python train_india.py
Output: saved_models/india_cnn.h5
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ── CONFIG ──────────────────────────────────────
DATA_PATH   = "data_india/Indian-Traffic Sign-Dataset/Images"
CSV_PATH    = "data_india/Indian-Traffic Sign-Dataset/traffic_sign.csv"
MODEL_SAVE  = "saved_models/india_cnn.h5"
IMG_SIZE    = 32
BATCH_SIZE  = 64
EPOCHS      = 35

# ── LOAD SIGN NAMES FROM CSV ─────────────────────
print("Loading sign names from CSV...")
csv_df    = pd.read_csv(CSV_PATH)
NUM_CLS   = len(csv_df)
sign_dict = dict(zip(csv_df["ClassId"], csv_df["Name"]))
print(f"Total classes in CSV: {NUM_CLS}")
print("Sample signs:", list(sign_dict.items())[:5])

# ── LOAD IMAGES ──────────────────────────────────
print("\nLoading images from", DATA_PATH)
X, y = [], []

class_folders = sorted(os.listdir(DATA_PATH), key=lambda x: int(x))

for folder in class_folders:
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    class_id = int(folder)
    files = os.listdir(folder_path)
    print(f"class {class_id:2d} ({sign_dict.get(class_id, '?')}): {len(files)} images")
    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
            continue
        try:
            path = os.path.join(folder_path, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(class_id)
        except Exception as e:
            print(f"  skipped {fname}: {e}")

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

# Re-map class IDs to 0..N-1
unique_cls = sorted(np.unique(y))
NUM_CLS    = len(unique_cls)
cls_map    = {old: new for new, old in enumerate(unique_cls)}
y_remap    = np.array([cls_map[c] for c in y])
y_enc      = to_categorical(y_remap, NUM_CLS)

# Save updated sign names (remapped)
sign_names_ordered = [sign_dict.get(c, f"Class {c}") for c in unique_cls]

print(f"\n✅ Total images loaded : {len(X)}")
print(f"✅ Actual classes      : {NUM_CLS}")

# ── TRAIN / VAL SPLIT (numpy — no sklearn needed) ─
np.random.seed(42)
idx    = np.random.permutation(len(X))
split  = int(len(X) * 0.8)
tr_idx = idx[:split]
v_idx  = idx[split:]
Xtr, Xv = X[tr_idx], X[v_idx]
ytr, yv  = y_enc[tr_idx], y_enc[v_idx]
print(f"\nTrain: {len(Xtr)}  Val: {len(Xv)}")

# ── BUILD CNN ────────────────────────────────────
def build_model(num_classes):
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

print("\nBuilding model...")
model = build_model(NUM_CLS)
model.summary()

# ── CALLBACKS ────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# ── TRAIN ────────────────────────────────────────
print("\n--- TRAINING INDIAN SIGN CNN ---")
history = model.fit(
    Xtr, ytr,
    validation_data=(Xv, yv),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── PLOT ─────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("India CNN Accuracy")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("India CNN Loss")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("india_training_history.png", dpi=120)
plt.show()

# ── SAVE SIGN NAMES ──────────────────────────────
import json
with open("saved_models/india_sign_names.json", "w") as f:
    json.dump(sign_names_ordered, f, indent=2)

print("\n" + "="*50)
print("✅ Training Complete!")
print(f"✅ Model saved  : {MODEL_SAVE}")
print(f"✅ Sign names   : saved_models/india_sign_names.json")
print(f"✅ Classes      : {NUM_CLS}")
print(f"✅ Signs        : {sign_names_ordered[:5]}...")
print("="*50)
print("\nTo use this model, run: python backend/app.py --model india")
