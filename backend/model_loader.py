import tensorflow as tf
import numpy as np
import cv2
import base64
import os

model = None

sign_names = [
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

sign_descriptions = [
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


def load_model(path):
    global model
    if not os.path.exists(path):
        print("model not found:", path)
        return False
    model = tf.keras.models.load_model(path)
    print("model loaded successfully:", path)
    return True


def preprocess(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    orig = img.copy()
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0), orig


def run_prediction(img_bytes):
    """Returns top-3 predictions as list of dicts."""
    inp, _ = preprocess(img_bytes)
    if inp is None:
        return None

    preds = model(inp, training=False).numpy()[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top3_idx:
        results.append({
            "class_id":    int(idx),
            "label":       sign_names[idx],
            "description": sign_descriptions[idx],
            "confidence":  round(float(preds[idx]) * 100, 2)
        })
    return results


def run_gradcam(img_bytes):
    """Returns base64-encoded Grad-CAM heatmap overlay PNG."""
    inp, orig = preprocess(img_bytes)
    if inp is None:
        return None

    # Find last conv layer
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name

    if last_conv_name is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(inp)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads      = tape.gradient(loss, conv_out)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out   = conv_out[0]
    heatmap    = conv_out @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap).numpy()
    heatmap    = np.maximum(heatmap, 0)
    mx         = heatmap.max()
    if mx > 0:
        heatmap /= mx

    h, w = orig.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored         = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay         = cv2.addWeighted(orig, 0.55, colored, 0.45, 0)

    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode("utf-8")
