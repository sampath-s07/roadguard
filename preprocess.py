import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


IMG_SIZE = 32
NUM_CLS = 43


def load_train(base):
    X = []
    y = []

    for cid in range(NUM_CLS):
        folder = os.path.join(base, "Train", str(cid))

        if not os.path.isdir(folder):
            print(f"folder missing for class {cid}")
            continue

        flist = os.listdir(folder)
        print(f"class {cid}: {len(flist)} images")

        for fname in flist:
            if not (fname.endswith(".png") or fname.endswith(".ppm") or fname.endswith(".jpg")):
                continue

            try:
                full = os.path.join(folder, fname)
                im = cv2.imread(full)
                if im is None:
                    continue

                im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                X.append(im)
                y.append(cid)

            except Exception as err:
                print("skipped:", fname, "->", err)

    print(f"\ntotal training images loaded: {len(X)}")
    return np.array(X), np.array(y)


def load_test(base):
    csv_file = os.path.join(base, "Test.csv")

    if not os.path.exists(csv_file):
        print("Test.csv not found in", base)
        return None, None

    df = pd.read_csv(csv_file)
    X = []
    y = []

    for _, row in df.iterrows():
        try:
            p = base + "/" + row["Path"]
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(row["ClassId"])
        except Exception as e:
            print("error:", e)

    print(f"test images loaded: {len(X)}")
    return np.array(X), np.array(y)


def norm(X):
    return X.astype("float32") / 255.0


def onehot(y):
    return to_categorical(y, num_classes=NUM_CLS)


def split_data(X, y):
    raw_y = np.argmax(y, axis=1)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y,
        test_size=0.2,
        stratify=raw_y,
        random_state=42
    )
    return Xtr, Xv, ytr, yv
