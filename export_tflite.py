"""
export_tflite.py
Converts both German and Indian trained Keras models to TensorFlow Lite format for Android/iOS.
Run: python export_tflite.py
"""

import tensorflow as tf
import os

MODELS = [
    {
        "name":   "German GTSRB (43 classes)",
        "h5":     os.path.join("saved_models", "custom_cnn.h5"),
        "tflite": os.path.join("saved_models", "custom_cnn.tflite"),
        "quant":  os.path.join("saved_models", "custom_cnn_quant.tflite")
    },
    {
        "name":   "Indian Signs (58 classes)",
        "h5":     os.path.join("saved_models", "india_cnn.h5"),
        "tflite": os.path.join("saved_models", "india_cnn.tflite"),
        "quant":  os.path.join("saved_models", "india_cnn_quant.tflite")
    }
]


def convert_model(h5_path, out_path, quantize=False):
    if not os.path.exists(h5_path):
        print(f"⚠️ Warning: Model not found at {h5_path}")
        return None

    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  -> Generated: {out_path} ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    print("=" * 60)
    print("RoadGuard — Mobile TensorFlow Lite Model Export")
    print("=" * 60)

    for item in MODELS:
        print(f"\nProcessing {item['name']}...")
        convert_model(item["h5"], item["tflite"], quantize=False)
        convert_model(item["h5"], item["quant"], quantize=True)

    print("\n✅ Mobile TFLite models exported successfully to saved_models/!")
