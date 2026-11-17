from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, json, time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_loader   import load_model, load_india_model, run_prediction, run_gradcam
from backend.yolo_detector  import load_yolo, detect_and_classify
from backend.video_processor import process_video

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GERMAN_MODEL  = os.path.join(BASE, "saved_models", "custom_cnn.h5")
INDIA_MODEL   = os.path.join(BASE, "saved_models", "india_cnn.h5")
INDIA_NAMES   = os.path.join(BASE, "saved_models", "india_sign_names.json")
YOLO_ONNX     = os.path.join(BASE, "saved_models", "yolov8n.onnx")
LOCATION_FILE = os.path.join(BASE, "saved_models", "sign_locations.json")

# Load all models at startup
ok_de   = load_model(GERMAN_MODEL)
ok_in   = load_india_model(INDIA_MODEL, INDIA_NAMES)
ok_yolo = load_yolo(YOLO_ONNX)

if not ok_de:   print("WARNING: German model load failed.")
if not ok_in:   print("WARNING: India model load failed.")
if not ok_yolo: print("WARNING: YOLOv8 not available.")


def get_locations():
    if os.path.exists(LOCATION_FILE):
        try:
            with open(LOCATION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_location_entry(entry):
    locations = get_locations()
    locations.insert(0, entry)
    # keep last 100 markers
    locations = locations[:100]
    with open(LOCATION_FILE, "w") as f:
        json.dump(locations, f, indent=2)
    return locations


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
        "models":  {"german": ok_de, "india": ok_in, "yolo": ok_yolo}
    })


# ── PREDICT ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    dataset = request.form.get("dataset", "german")
    lat_val = request.form.get("lat")
    lng_val = request.form.get("lng")

    results = run_prediction(f.read(), dataset=dataset)
    if results is None:
        return jsonify({"error": "could not process image"}), 500

    top = results[0]
    print(f"[{dataset}] {top['label']} ({top['confidence']}%)")

    # Geotag if location provided
    if lat_val and lng_val:
        try:
            entry = {
                "id":         f"loc_{int(time.time()*1000)}",
                "label":      top["label"],
                "confidence": top["confidence"],
                "dataset":    dataset,
                "lat":        float(lat_val),
                "lng":        float(lng_val),
                "time":       time.strftime("%I:%M %p"),
                "type":       "danger" if any(w in top["label"].lower() for w in ["stop", "no", "danger", "warning", "curve"]) else "standard"
            }
            save_location_entry(entry)
        except Exception as e:
            print("Location save error:", e)

    return jsonify({"top": results, "dataset": dataset})


# ── DETECT (YOLOv8) ────────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if not ok_yolo:
        return jsonify({"error": "YOLOv8 ONNX not loaded on server"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400

    dataset = request.form.get("dataset", "german")
    b64, detections = detect_and_classify(
        request.files["file"].read(), run_prediction, dataset=dataset)
    if b64 is None:
        return jsonify({"error": "detection failed"}), 500

    print(f"[yolo/{dataset}] {len(detections)} object(s)")
    return jsonify({"image": b64, "detections": detections,
                    "count": len(detections), "dataset": dataset})


# ── VIDEO PROCESSING ───────────────────────────────────────────────────────────
@app.route("/process_video", methods=["POST"])
def process_video_route():
    if "file" not in request.files:
        return jsonify({"error": "no video file"}), 400

    f       = request.files["file"]
    dataset = request.form.get("dataset", "german")

    video_bytes = f.read()
    if len(video_bytes) > 50 * 1024 * 1024:
        return jsonify({"error": "Video too large (max 50 MB)"}), 413

    result = process_video(video_bytes, run_prediction, dataset=dataset)
    if "error" in result:
        return jsonify(result), 500

    print(f"[video/{dataset}] {result['sampled']} frames, "
          f"{len(result['summary'])} unique signs")
    return jsonify(result)


# ── PREDICT BATCH ──────────────────────────────────────────────────────────────
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    dataset = request.form.get("dataset", "german")
    output  = []
    for f in files:
        results = run_prediction(f.read(), dataset=dataset)
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
    b64     = run_gradcam(f.read(), dataset=dataset)
    if b64 is None:
        return jsonify({"error": "gradcam failed"}), 500
    return jsonify({"image": b64})


# ── GPS & MAP API ──────────────────────────────────────────────────────────────
@app.route("/api/locations", methods=["GET"])
def get_all_locations():
    return jsonify(get_locations())

@app.route("/api/locations", methods=["POST"])
def add_location():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data or "lat" not in data or "lng" not in data or "label" not in data:
        return jsonify({"error": "Missing lat, lng, or label"}), 400
    try:
        entry = {
            "id":         f"loc_{int(time.time()*1000)}",
            "label":      data["label"],
            "confidence": float(data.get("confidence", 95.0)),
            "dataset":    data.get("dataset", "german"),
            "lat":        float(data["lat"]),
            "lng":        float(data["lng"]),
            "time":       data.get("time", time.strftime("%I:%M %p")),
            "type":       data.get("type", "standard")
        }
        updated = save_location_entry(entry)
        return jsonify({"success": True, "location": entry, "count": len(updated)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/locations", methods=["DELETE"])
def clear_locations():
    with open(LOCATION_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True, "message": "Locations cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
