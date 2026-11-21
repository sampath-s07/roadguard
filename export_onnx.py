# ───────────────────────────────────────────────────
# export_onnx.py
# Run this in Google Colab or any machine where TF works
# to export .h5 models → .onnx for local Windows use
# ───────────────────────────────────────────────────
# Colab steps:
#   1. Upload custom_cnn.h5 and india_cnn.h5 to Colab
#   2. Run:  !pip install tf2onnx
#   3. Run this script
#   4. Download custom_cnn.onnx and india_cnn.onnx
#   5. Put them in roadgaurd/saved_models/
# ───────────────────────────────────────────────────

import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

def export(h5_path, onnx_path, input_shape=(1, 32, 32, 3)):
    print(f"Loading {h5_path} ...")
    model = tf.keras.models.load_model(h5_path, compile=False)
    model.summary()

    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    print(f"Converting to ONNX → {onnx_path} ...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                                  opset=13, output_path=onnx_path)
    print(f"✅ Saved: {onnx_path}  ({onnx_proto_size(onnx_path):.1f} MB)")

def onnx_proto_size(path):
    return os.path.getsize(path) / (1024*1024)

import os
BASE = os.path.dirname(os.path.abspath(__file__))

export(
    os.path.join(BASE, "saved_models", "custom_cnn.h5"),
    os.path.join(BASE, "saved_models", "custom_cnn.onnx")
)

export(
    os.path.join(BASE, "saved_models", "india_cnn.h5"),
    os.path.join(BASE, "saved_models", "india_cnn.onnx")
)

print("\n✅ Both models exported. Download custom_cnn.onnx and india_cnn.onnx")
print("   Put them in: roadgaurd/saved_models/")
