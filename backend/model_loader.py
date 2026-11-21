"""
model_loader.py  –  HIGH-LEVEL v4
Local: uses onnxruntime (bypasses blocked TF DLLs on Windows App Control machines)
Render/Cloud: falls back to TensorFlow (full accuracy, all features)

Priority order:
  1. ONNX Runtime (.onnx exported models)  ← local Windows
  2. TFLite Runtime (.tflite)              ← mobile / edge
  3. TensorFlow Keras (.h5)                ← cloud / Render
"""

import numpy as np
import cv2
import base64
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="[ModelLoader] %(message)s")
log = logging.getLogger(__name__)

# ── Runtime detection ─────────────────────────────────────────────────────────
_HAS_ORT = False
_HAS_TF  = False
_HAS_TFL = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
    log.info(f"onnxruntime {ort.__version__} available")
except Exception as e:
    log.info(f"onnxruntime not available: {e}")

try:
    import tensorflow as tf
    _HAS_TF = True
    log.info(f"TensorFlow {tf.__version__} available")
except Exception as e:
    log.warning(f"TensorFlow not available: {e}")

try:
    import tflite_runtime.interpreter as tflite
    _HAS_TFL = True
    log.info("tflite_runtime available")
except Exception:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE          = 32
CONF_REJECT_THOLD = 15.0
TTA_ENABLED       = True

# ── Model handles ─────────────────────────────────────────────────────────────
# ORT sessions
_de_ort  = None
_in_ort  = None
# TF Keras models
_de_tf   = None
_in_tf   = None
# TFLite interpreters
_de_tfl  = None
_in_tfl  = None

india_sign_names = []

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


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _enhance(img_bgr):
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(img_bytes, enhance=True):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    orig = img.copy()
    if enhance:
        img = _enhance(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0), orig


# ── Inference helpers ─────────────────────────────────────────────────────────
def _infer_ort(session, inp):
    name = session.get_inputs()[0].name
    return session.run(None, {name: inp})[0][0]


def _infer_tf(model, inp):
    if TTA_ENABLED:
        variants = np.stack([
            inp[0],
            inp[0, :, ::-1, :],
            inp[0, ::-1, :, :],
            np.clip(inp[0] * 1.1, 0, 1)
        ], axis=0)
        return np.mean(model(variants, training=False).numpy(), axis=0)
    return model(inp, training=False).numpy()[0]


def _infer_tfl(interp, inp):
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    interp.set_tensor(in_idx, inp.astype("float32"))
    interp.invoke()
    return interp.get_tensor(out_idx)[0]


# ── Load functions ────────────────────────────────────────────────────────────
def load_model(h5_path):
    """Load German CNN. Tries ORT → TFLite → TF in order."""
    global _de_ort, _de_tf, _de_tfl

    onnx_path  = h5_path.replace(".h5", ".onnx")
    tflite_path = h5_path.replace(".h5", ".tflite")

    if _HAS_ORT and os.path.exists(onnx_path):
        try:
            _de_ort = ort.InferenceSession(onnx_path,
                        providers=["CPUExecutionProvider"])
            log.info(f"German ORT session loaded: {onnx_path}")
            return True
        except Exception as e:
            log.warning(f"ORT load failed: {e}")

    if _HAS_TFL and os.path.exists(tflite_path):
        try:
            interp = tflite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            _de_tfl = interp
            log.info(f"German TFLite loaded: {tflite_path}")
            return True
        except Exception as e:
            log.warning(f"TFLite load failed: {e}")

    if _HAS_TF and os.path.exists(h5_path):
        try:
            _de_tf = tf.keras.models.load_model(h5_path, compile=False)
            dummy  = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
            _de_tf(dummy, training=False)
            log.info(f"German TF model loaded: {h5_path}")
            return True
        except Exception as e:
            log.error(f"TF load failed: {e}")

    log.error("German model: no runtime available or file missing.")
    return False


def load_india_model(model_path, names_path):
    """Load Indian CNN. Tries ORT → TFLite → TF in order."""
    global _in_ort, _in_tf, _in_tfl, india_sign_names

    if not os.path.exists(names_path):
        log.error(f"India sign names not found: {names_path}")
        return False
    with open(names_path) as f:
        india_sign_names = json.load(f)

    onnx_path   = model_path.replace(".h5", ".onnx")
    tflite_path = model_path.replace(".h5", ".tflite")

    if _HAS_ORT and os.path.exists(onnx_path):
        try:
            _in_ort = ort.InferenceSession(onnx_path,
                        providers=["CPUExecutionProvider"])
            log.info(f"India ORT session loaded: {onnx_path} ({len(india_sign_names)} classes)")
            return True
        except Exception as e:
            log.warning(f"India ORT load failed: {e}")

    if _HAS_TFL and os.path.exists(tflite_path):
        try:
            interp = tflite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            _in_tfl = interp
            log.info(f"India TFLite loaded: {tflite_path}")
            return True
        except Exception as e:
            log.warning(f"India TFLite load failed: {e}")

    if _HAS_TF and os.path.exists(model_path):
        try:
            _in_tf = tf.keras.models.load_model(model_path, compile=False)
            dummy  = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
            _in_tf(dummy, training=False)
            log.info(f"India TF model loaded: {model_path} ({len(india_sign_names)} classes)")
            return True
        except Exception as e:
            log.error(f"India TF load failed: {e}")

    log.error("India model: no runtime available or file missing.")
    return False


# ── Prediction ────────────────────────────────────────────────────────────────
def run_prediction(img_bytes, dataset="german"):
    inp, _ = preprocess(img_bytes, enhance=True)
    if inp is None:
        return None

    if dataset == "india":
        if   _in_ort is not None: preds = _infer_ort(_in_ort, inp)
        elif _in_tf  is not None: preds = _infer_tf(_in_tf, inp)
        elif _in_tfl is not None: preds = _infer_tfl(_in_tfl, inp)
        else: return None
        names = india_sign_names
        descs = None
    else:
        if   _de_ort is not None: preds = _infer_ort(_de_ort, inp)
        elif _de_tf  is not None: preds = _infer_tf(_de_tf, inp)
        elif _de_tfl is not None: preds = _infer_tfl(_de_tfl, inp)
        else: return None
        names = german_sign_names
        descs = german_sign_descriptions

    top3_idx = np.argsort(preds)[::-1][:3]
    results  = []
    for rank, idx in enumerate(top3_idx):
        label = names[idx] if idx < len(names) else f"Class {idx}"
        desc  = descs[idx] if (descs and idx < len(descs)) else f"Indian traffic sign: {label}"
        conf  = round(float(preds[idx]) * 100, 2)
        if rank == 0 and conf < CONF_REJECT_THOLD:
            label = "Low confidence / unclear sign"
            desc  = "Image may be blurry, poorly lit, or not a traffic sign."
        results.append({"class_id": int(idx), "label": label,
                        "description": desc, "confidence": conf})
    return results


# ── Grad-CAM (TF only) ────────────────────────────────────────────────────────
def run_gradcam(img_bytes, dataset="german"):
    if not _HAS_TF:
        return None   # Grad-CAM needs TF GradientTape

    inp, orig = preprocess(img_bytes, enhance=True)
    if inp is None:
        return None

    mdl = _in_tf if dataset == "india" else _de_tf
    if mdl is None:
        return None

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

    grads    = tape.gradient(loss, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(1, 2))[0]
    heatmap  = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
    heatmap  = np.maximum(heatmap, 0)
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx

    h, w = orig.shape[:2]
    hmap  = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.60, colored, 0.40, 0)

    label = german_sign_names[top_class] if (dataset != "india" and top_class < len(german_sign_names)) \
            else (india_sign_names[top_class] if top_class < len(india_sign_names) else f"Class {top_class}")
    cv2.putText(overlay, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode("utf-8")


# ── Transfer model builder (Render/Colab only) ────────────────────────────────
def build_transfer_model(num_classes, base="efficientnetb0"):
    if not _HAS_TF:
        raise RuntimeError("TensorFlow required to build transfer model")
    inp = tf.keras.Input(shape=(96, 96, 3))
    if base == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(96,96,3), include_top=False, weights="imagenet", pooling="avg")
    else:
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(96,96,3), include_top=False, weights="imagenet", pooling="avg")
    backbone.trainable = False
    x = backbone(inp, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"RoadGuard_{base}_{num_classes}cls")
