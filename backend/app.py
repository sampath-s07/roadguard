from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_loader  import load_model, load_india_model, run_prediction, run_gradcam
from backend.yolo_detector import load_yolo, detect_and_classify

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GERMAN_MODEL = os.path.join(BASE, "saved_models", "custom_cnn.h5")
INDIA_MODEL  = os.path.join(BASE, "saved_models", "india_cnn.h5")
INDIA_NAMES  = os.path.join(BASE, "saved_models", "india_sign_names.json")

# Load models at startup
ok_de   = load_model(GERMAN_MODEL)
ok_in   = load_india_model(INDIA_MODEL, INDIA_NAMES)
ok_yolo = load_yolo()

if not ok_de:   print("WARNING: German model load failed.")
if not ok_in:   print("WARNING: India model load failed (optional).")
if not ok_yolo: print("WARNING: YOLOv8 not available (pip install ultralytics).")


# ── PAGES ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/gallery")
def gallery():
    return send_from_directory(app.static_folder, "gallery.html")


# ── STATUS ─────────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "status":  "RoadGuard API running",
        "version": "2.0",
        "models": {
            "german": ok_de,
            "india":  ok_in,
            "yolo":   ok_yolo
        }
    })


# ── PREDICT (single image, classify) ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    dataset = request.form.get("dataset", "german")
    raw     = f.read()
    results = run_prediction(raw, dataset=dataset)
    if results is None:
        return jsonify({"error": "could not process image"}), 500

    top = results[0]
    print(f"[{dataset}] predicted: {top['label']} ({top['confidence']}%)")
    return jsonify({"top": results, "dataset": dataset})


# ── DETECT (YOLOv8 bounding boxes + CNN classify) ─────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if not ok_yolo:
        return jsonify({"error": "YOLOv8 not installed on server. Run: pip install ultralytics"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400

    f       = request.files["file"]
    dataset = request.form.get("dataset", "german")
    raw     = f.read()

    b64, detections = detect_and_classify(raw, run_prediction, dataset=dataset)
    if b64 is None:
        return jsonify({"error": "detection failed"}), 500

    print(f"[yolo/{dataset}] {len(detections)} object(s) detected")
    return jsonify({
        "image":      b64,          # annotated image base64
        "detections": detections,   # list of boxes + labels
        "count":      len(detections),
        "dataset":    dataset
    })


# ── PREDICT BATCH ──────────────────────────────────────────────────────────────
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    dataset = request.form.get("dataset", "german")
    output  = []
    for f in files:
        raw     = f.read()
        results = run_prediction(raw, dataset=dataset)
        if results is None:
            output.append({"filename": f.filename, "error": "could not process"})
        else:
            top = results[0]
            output.append({
                "filename":    f.filename,
                "label":       top["label"],
                "description": top["description"],
                "confidence":  top["confidence"],
                "top3":        results
            })
    return jsonify({"results": output, "dataset": dataset})


# ── GRAD-CAM ───────────────────────────────────────────────────────────────────
@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400
    f       = request.files["file"]
    dataset = request.form.get("dataset", "german")
    raw     = f.read()
    b64     = run_gradcam(raw, dataset=dataset)
    if b64 is None:
        return jsonify({"error": "gradcam failed"}), 500
    return jsonify({"image": b64})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
