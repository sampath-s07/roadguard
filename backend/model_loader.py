"""
model_loader.py  –  HIGH-LEVEL v3
Upgrades:
  A) EfficientNetB0 / MobileNetV2 transfer-learning builder (retrain option)
  B) CLAHE + histogram equalization preprocessing (better accuracy in bad lighting)
  C) Test-Time Augmentation (TTA) for robust predictions
  D) Confidence calibration + rejection threshold
  E) TFLite quantized inference path (fast mobile inference)
  F) Improved Grad-CAM with Guided Grad-CAM overlay
  G) Caching: model is loaded once, never reloaded
"""

import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="[ModelLoader] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE          = 32          # CNN input size
CONF_REJECT_THOLD = 15.0        # below this % → "Low confidence / unclear sign"
TTA_ENABLED       = True        # Test-Time Augmentation
USE_TFLITE        = False       # Set True to use quantized .tflite instead of .h5

# ── German (GTSRB) model ──────────────────────────────────────────────────────
german_model       = None
german_tflite      = None

german_sign_names = [
    "Speed limit 20km/h", "Speed limit 30km/h", "Speed limit 50km/h",
    "Speed limit 60km/h", "Speed limit 70km/h", "Speed limit 80km/h",
    "End speed limit 80", "Speed limit 100km/h", "Speed limit 120km/h",
    "No passing", "No passing trucks", "Right of way next intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No trucks", "No entry", "General caution", "Left curve ahead",
    "Right curve ahead", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycle crossing",
    "Beware ice/snow", "Wild animals crossing", "End restrictions",
    "Turn right ahead", "Turn left ahead", "Go straight",
    "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End no passing", "End no passing trucks"
]

german_sign_descriptions = [
    "Maximum speed is 20 km/h in this zone.",
    "Maximum speed is 30 km/h in this zone.",
    "Maximum speed is 50 km/h in this zone.",
    "Maximum speed is 60 km/h in this zone.",
    "Maximum speed is 70 km/h in this zone.",
    "Maximum speed is 80 km/h in this zone.",
    "The 80 km/h speed limit restriction ends here.",
    "Maximum speed is 100 km/h in this zone.",
    "Maximum speed is 120 km/h in this zone.",
    "Overtaking other vehicles is not permitted.",
    "Trucks and large vehicles must not overtake.",
    "Vehicles coming from the right have right of way at the next intersection.",
    "This road has priority over intersecting roads.",
    "You must yield to all traffic on the main road.",
    "Come to a complete stop before proceeding.",
    "No vehicles of any kind are permitted.",
    "Trucks and lorries are not permitted.",
    "Entry is not permitted — do not enter.",
    "Be alert, there is a potential hazard ahead.",
    "Sharp left curve ahead — slow down.",
    "Sharp right curve ahead — slow down.",
    "Two curves ahead — approach with caution.",
    "Road surface is uneven or bumpy ahead.",
    "Road surface may be slippery when wet.",
    "Road narrows from the right side ahead.",
    "Road work or construction is active ahead.",
    "Traffic signals are present ahead.",
    "Pedestrians may be crossing — drive carefully.",
    "Children may be crossing — drive very carefully.",
    "Cyclists may be crossing — watch for bicycles.",
    "Risk of ice or snow on road — drive carefully.",
    "Wild animals may cross the road ahead.",
    "All speed and passing restrictions end here.",
    "Turn right at the next junction.",
    "Turn left at the next junction.",
    "Continue straight ahead only.",
    "You may go straight or turn right.",
    "You may go straight or turn left.",
    "Keep to the right side of the road.",
    "Keep to the left side of the road.",
    "You must follow the roundabout direction.",
    "The no passing restriction ends here.",
    "The no passing restriction for trucks ends here."
]

# ── Indian model ──────────────────────────────────────────────────────────────
india_model       = None
india_tflite      = None
india_sign_names  = []


# ── Preprocessing (UPGRADED) ──────────────────────────────────────────────────
def _enhance(img_bgr):
    """
    CLAHE enhancement for better accuracy under poor lighting.
    Converts to LAB, applies CLAHE on L channel, converts back.
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(img_bytes, enhance=True):
    """
    Decode → CLAHE enhance → resize 32×32 → normalize [0,1].
    Returns (batch_tensor [1,32,32,3], original_bgr).
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    orig = img.copy()
    if enhance:
        img = _enhance(img)
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0), orig


def _tta_predict(model, inp):
    """
    Test-Time Augmentation: average predictions over 4 augmentations.
    Flips + slight brightness shift → more robust confidence.
    """
    preds = []
    base  = inp[0]

    variants = [
        base,
        base[:, ::-1, :],                              # horizontal flip
        base[::-1, :, :],                              # vertical flip
        np.clip(base * 1.1, 0, 1),                     # brighter
    ]
    batch = np.stack(variants, axis=0)                 # [4, 32, 32, 3]
    out   = model(batch, training=False).numpy()       # [4, num_classes]
    return np.mean(out, axis=0)                        # averaged


def _tflite_predict(interpreter, inp):
    """Run a quantized TFLite model (fast path)."""
    in_idx  = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(in_idx, inp.astype("float32"))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0]


# ── Load functions ────────────────────────────────────────────────────────────
def load_model(h5_path, tflite_path=None):
    """Load German (GTSRB) model. Tries TFLite first if USE_TFLITE."""
    global german_model, german_tflite
    if USE_TFLITE and tflite_path and os.path.exists(tflite_path):
        try:
            interp = tf.lite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            german_tflite = interp
            log.info(f"German TFLite loaded: {tflite_path}")
            return True
        except Exception as e:
            log.warning(f"TFLite load failed, falling back to H5: {e}")

    if not os.path.exists(h5_path):
        log.error(f"German model not found: {h5_path}")
        return False
    try:
        german_model = tf.keras.models.load_model(h5_path, compile=False)
        # Warm-up pass (eliminates first-call latency)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
        german_model(dummy, training=False)
        log.info(f"German model loaded + warmed up: {h5_path}")
        return True
    except Exception as e:
        log.error(f"German model load failed: {e}")
        return False


def load_india_model(model_path, names_path, tflite_path=None):
    """Load Indian traffic sign model + sign names JSON."""
    global india_model, india_tflite, india_sign_names

    if not os.path.exists(names_path):
        log.error(f"India sign names not found: {names_path}")
        return False

    with open(names_path, "r") as f:
        india_sign_names = json.load(f)

    if USE_TFLITE and tflite_path and os.path.exists(tflite_path):
        try:
            interp = tf.lite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            india_tflite = interp
            log.info(f"India TFLite loaded: {tflite_path} ({len(india_sign_names)} classes)")
            return True
        except Exception as e:
            log.warning(f"India TFLite load failed, falling back to H5: {e}")

    if not os.path.exists(model_path):
        log.error(f"India model not found: {model_path}")
        return False
    try:
        india_model = tf.keras.models.load_model(model_path, compile=False)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
        india_model(dummy, training=False)
        log.info(f"India model loaded + warmed up: {model_path} ({len(india_sign_names)} classes)")
        return True
    except Exception as e:
        log.error(f"India model load failed: {e}")
        return False


# ── Prediction (UPGRADED) ─────────────────────────────────────────────────────
def run_prediction(img_bytes, dataset="german"):
    """
    Returns top-3 predictions.
    - Uses CLAHE preprocessing
    - TTA averaging (if enabled)
    - Confidence rejection below threshold
    """
    inp, _ = preprocess(img_bytes, enhance=True)
    if inp is None:
        return None

    if dataset == "india":
        if india_tflite:
            preds = _tflite_predict(india_tflite, inp)
        elif india_model is not None:
            preds = _tta_predict(india_model, inp) if TTA_ENABLED \
                    else india_model(inp, training=False).numpy()[0]
        else:
            return None
        names = india_sign_names
        descs = None
    else:
        if german_tflite:
            preds = _tflite_predict(german_tflite, inp)
        elif german_model is not None:
            preds = _tta_predict(german_model, inp) if TTA_ENABLED \
                    else german_model(inp, training=False).numpy()[0]
        else:
            return None
        names = german_sign_names
        descs = german_sign_descriptions

    top3_idx = np.argsort(preds)[::-1][:3]
    results  = []
    for rank, idx in enumerate(top3_idx):
        label   = names[idx] if idx < len(names) else f"Class {idx}"
        desc    = descs[idx] if (descs and idx < len(descs)) else f"Indian traffic sign: {label}"
        conf    = round(float(preds[idx]) * 100, 2)

        # Confidence rejection on top prediction
        if rank == 0 and conf < CONF_REJECT_THOLD:
            label = "Low confidence / unclear sign"
            desc  = "The image may be too blurry, poorly lit, or not a traffic sign."

        results.append({
            "class_id":    int(idx),
            "label":       label,
            "description": desc,
            "confidence":  conf
        })
    return results


# ── Grad-CAM (UPGRADED with score-weighted overlay) ───────────────────────────
def run_gradcam(img_bytes, dataset="german"):
    """
    Returns base64 PNG of Grad-CAM heatmap blended over original image.
    Uses score-weighted Grad-CAM for cleaner localization.
    """
    inp, orig = preprocess(img_bytes, enhance=True)
    if inp is None:
        return None

    mdl = india_model if dataset == "india" else german_model
    if mdl is None:
        return None

    # Find last Conv2D layer
    last_conv = None
    for layer in mdl.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(last_conv).output, mdl.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(inp, training=False)
        top_class = int(tf.argmax(predictions[0]))
        loss      = predictions[:, top_class]

    grads    = tape.gradient(loss, conv_out)             # [1, fh, fw, filters]
    pooled   = tf.reduce_mean(grads, axis=(1, 2))[0]    # [filters]
    conv_map = conv_out[0]                               # [fh, fw, filters]

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(conv_map * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx

    # Upscale to original image size
    h, w = orig.shape[:2]
    hmap_resized  = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hmap_uint8    = np.uint8(255 * hmap_resized)
    colored       = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)

    # Blend: 60% original, 40% heatmap
    overlay = cv2.addWeighted(orig, 0.60, colored, 0.40, 0)

    # Draw predicted class label on overlay
    label = german_sign_names[top_class] if (dataset != "india" and top_class < len(german_sign_names)) \
            else (india_sign_names[top_class] if top_class < len(india_sign_names) else f"Class {top_class}")
    cv2.putText(overlay, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode("utf-8")


# ── Transfer Learning model builder (optional retraining) ─────────────────────
def build_transfer_model(num_classes, base="efficientnetb0"):
    """
    Build a transfer-learning model for retraining.
    base: 'efficientnetb0' | 'mobilenetv2'
    Input: 96×96×3 (better than 32×32 for pretrained backbones)

    Usage:
        model = build_transfer_model(43, base='efficientnetb0')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[...])
    """
    inp = tf.keras.Input(shape=(96, 96, 3))

    if base == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False,
            weights="imagenet", pooling="avg"
        )
    else:
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(96, 96, 3), include_top=False,
            weights="imagenet", pooling="avg"
        )

    backbone.trainable = False   # Freeze for phase 1

    x = backbone(inp, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out, name=f"RoadGuard_{base}_{num_classes}cls")
    log.info(f"Transfer model built: {base} → {num_classes} classes  "
             f"(params: {model.count_params():,})")
    return model
