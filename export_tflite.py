"""
export_tflite.py
Converts the trained Keras model to TensorFlow Lite format.
Run: python export_tflite.py
Output: saved_models/custom_cnn.tflite  (~400KB, optimized)
"""

import tensorflow as tf
import os

MODEL_PATH  = os.path.join("saved_models", "custom_cnn.h5")
TFLITE_PATH = os.path.join("saved_models", "custom_cnn.tflite")
TFLITE_Q    = os.path.join("saved_models", "custom_cnn_quant.tflite")


def export(quantize=False):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run main.py first to train and save the model.")
        return

    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. Converting to TFLite ...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # Dynamic range quantization — 4x smaller, slightly lower accuracy
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        out_path = TFLITE_Q
    else:
        out_path = TFLITE_PATH

    tflite_model = converter.convert()

    os.makedirs("saved_models", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n✅ Saved: {out_path}")
    print(f"   Size : {size_kb:.1f} KB")
    print("\nUse this file on Android / iOS / Raspberry Pi!")
    return out_path


if __name__ == "__main__":
    print("=" * 50)
    print("RoadGuard — TensorFlow Lite Export")
    print("=" * 50)

    # Standard export
    export(quantize=False)

    # Quantized (smaller) export
    print("\nAlso creating quantized version (smaller) ...")
    export(quantize=True)

    orig_kb  = os.path.getsize(TFLITE_PATH)  / 1024
    quant_kb = os.path.getsize(TFLITE_Q)     / 1024
    h5_kb    = os.path.getsize(MODEL_PATH)   / 1024

    print("\n" + "=" * 50)
    print(f"{'Model':<30} {'Size':>10}")
    print("-" * 40)
    print(f"{'custom_cnn.h5 (original)':<30} {h5_kb:>8.0f} KB")
    print(f"{'custom_cnn.tflite':<30} {orig_kb:>8.0f} KB")
    print(f"{'custom_cnn_quant.tflite':<30} {quant_kb:>8.0f} KB")
    print("=" * 50)
