import os
import random
import numpy as np
import tensorflow as tf


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

from preprocess import load_train, load_test, norm, onehot, split_data
from models import build_cnn, build_transfer
from train import run_training
from evaluate import test_model, plot_history, plot_cm, show_preds, bar_chart

DATA = "./data"

if not os.path.exists(DATA + "/Train"):
    print("dataset not found in ./data/ folder")
    print("please download GTSRB from kaggle and place in ./data/")
    exit()
else:
    print("dataset found, loading...")

print("\nloading images...")
x, y =load_train(DATA)
xt, yt = load_test(DATA)

if xt is None:
    print("test data load failed, check folder structure")
    exit()

x = norm(x)
xt = norm(xt)

y_enc = onehot(y)
yt_enc = onehot(yt)

xtr, xv, ytr, yv = split_data(x, y_enc)
print(f"train: {len(xtr)} val: {len(xv)} test: {len(xt)}")

results = {}

print("\n--- CUSTOM CNN ---")
cnn = build_cnn()
cnn.summary()

h1 = run_training(cnn, xtr, ytr, xv, yv, name = "custom_cnn", epochs = 35, bs = 64)

plot_history(h1, "CustomCNN")
yt_true, yt_pred, acc1 = test_model(cnn, xt, yt_enc)
plot_cm(yt_true, yt_pred, "CustomCNN")
show_preds(xt, yt_true, yt_pred)
results["custom CNN"] = acc1


print("\n--- MOBILENETV2 ---")
mob, mob_base = build_transfer()
mob.summary()

h2 = run_training(mob, xtr, ytr, xv, yv, name="mobilenet", epochs = 25, bs = 64)

plot_history(h2, "MobileNetV2")
yt_true2, yt_pred2, acc2 = test_model(mob, xt, yt_enc)
plot_cm(yt_true2, yt_pred2, "MobileNetV2")
show_preds(xt, yt_true2, yt_pred2)
results["MobileNetV2"] = acc2

bar_chart(results)

print("\nfinal results:")
for k, v in results.items():
    print(f"  {k} -> {v * 100:.2f}%")