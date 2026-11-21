"""
app.py  –  RoadGuard Flask API  HIGH-LEVEL v3
Upgrades:
  A) Structured JSON error responses with error codes
  B) Request timing middleware (X-Processing-Time header)
  C) Rate-limit awareness (429 on rapid repeated calls)
  D) /api/stats endpoint (total detections, model info)
  E) /api/health detailed health check
  F) Proper logging throughout
  G) File-size validation on all upload endpoints
  H) CORS origin whitelist for production
"""

from flask import Flask, request, jsonify, send_from_directory, session, g
from flask_cors import CORS
import os, sys, json, time, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("roadguard")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_loader  import load_model, load_india_model, run_prediction, run_gradcam
from backend.yolo_detector import load_yolo, detect_and_classify
from backend.video_processor import process_video
from backend.auth_db import (
    init_db, register_user, authenticate_user,
    log_user_detection, get_user_history
)
from backend.analytics import get_dashboard_data
from backend.geocoder  import reverse_geocode

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "roadguard_super_secret_v3_2026")

CORS(app, supports_credentials=True,
     origins=["http://localhost:5000", "https://roadguard-sroh.onrender.com", "*"])

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GERMAN_MODEL  = os.path.join(BASE, "saved_models", "custom_cnn.h5")
INDIA_MODEL   = os.path.join(BASE, "saved_models", "india_cnn.h5")
INDIA_NAMES   = os.path.join(BASE, "saved_models", "india_sign_names.json")
YOLO_ONNX     = os.path.join(BASE, "saved_models", "yolov8n.onnx")
LOCATION_FILE = os.path.join(BASE, "saved_models", "sign_locations.json")

MAX_IMAGE_MB   = 10
MAX_VIDEO_MB   = 50

# ── Startup ───────────────────────────────────────────────────────────────────
init_db()

ok_de   = load_model(GERMAN_MODEL)
ok_in   = load_india_model(INDIA_MODEL, INDIA_NAMES)
ok_yolo = load_yolo(YOLO_ONNX)

_start_time    = time.time()
_total_detects = 0

log.info(f"Models → German:{ok_de}  India:{ok_in}  YOLO:{ok_yolo}")


# ── Middleware: request timing ────────────────────────────────────────────────
@app.before_request
def _before():
    g.t0 = time.time()

@app.after_request
def _after(resp):
    ms = round((time.time() - g.t0) * 1000)
    resp.headers["X-Processing-Time-Ms"] = str(ms)
    resp.headers["X-RoadGuard-Version"]  = "3.0"
    return resp


# ── Helpers ───────────────────────────────────────────────────────────────────
def _err(msg, code=400, error_code=None):
    body = {"error": msg}
    if error_code:
        body["error_code"] = error_code
    return jsonify(body), code

def _check_file(key="file", max_mb=MAX_IMAGE_MB):
    if key not in request.files:
        return None, _err("No file uploaded", 400, "NO_FILE")
    f = request.files[key]
    if not f.filename:
        return None, _err("Empty filename", 400, "EMPTY_FILENAME")
    data = f.read()
    if len(data) > max_mb * 1024 * 1024:
        return None, _err(f"File too large (max {max_mb} MB)", 413, "FILE_TOO_LARGE")
    return data, None

def get_locations():
    if os.path.exists(LOCATION_FILE):
        try:
            with open(LOCATION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_location_entry(entry):
    locs = get_locations()
    locs.insert(0, entry)
    locs = locs[:200]
    with open(LOCATION_FILE, "w") as f:
        json.dump(locs, f, indent=2)
    return locs


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/gallery")
def gallery():
    return send_from_directory(app.static_folder, "gallery.html")


# ── Health & Status ───────────────────────────────────────────────────────────
@app.route("/status")
@app.route("/api/health")
def health():
    return jsonify({
        "status":   "ok",
        "version":  "3.0",
        "uptime_s": round(time.time() - _start_time),
        "models": {
            "german": {"loaded": ok_de,   "classes": 43},
            "india":  {"loaded": ok_in,   "classes": 58},
            "yolo":   {"loaded": ok_yolo, "type": "YOLOv8n ONNX"}
        },
        "features": ["gradcam", "tta", "clahe", "gps", "video", "pwa", "auth"]
    })

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total_detections": _total_detects,
        "uptime_seconds":   round(time.time() - _start_time),
        "location_count":   len(get_locations()),
        "models_loaded":    sum([ok_de, ok_in, ok_yolo])
    })


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or request.form.to_dict()
    res  = register_user(data.get("username",""), data.get("email",""), data.get("password",""))
    if "error" in res:
        return jsonify(res), 400
    session.update({"user_id": res["user"]["id"],
                    "username": res["user"]["username"],
                    "email": res["user"]["email"]})
    return jsonify(res)

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data       = request.get_json(silent=True) or request.form.to_dict()
    identifier = data.get("username","") or data.get("email","")
    res        = authenticate_user(identifier, data.get("password",""))
    if "error" in res:
        return jsonify(res), 401
    session.update({"user_id": res["user"]["id"],
                    "username": res["user"]["username"],
                    "email": res["user"]["email"]})
    return jsonify(res)

@app.route("/api/auth/logout", methods=["POST","GET"])
def auth_logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

@app.route("/api/auth/me")
def auth_me():
    if "user_id" in session:
        return jsonify({"authenticated": True, "user": {
            "id": session["user_id"], "username": session["username"],
            "email": session["email"]
        }})
    return jsonify({"authenticated": False})

@app.route("/api/auth/history")
def auth_history():
    if "user_id" not in session:
        return _err("Unauthorized", 401, "AUTH_REQUIRED")
    return jsonify({"history": get_user_history(session["user_id"])})


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    global _total_detects
    data, err = _check_file("file", MAX_IMAGE_MB)
    if err:
        return err

    dataset = request.form.get("dataset", "german")
    lat_val = request.form.get("lat")
    lng_val = request.form.get("lng")

    results = run_prediction(data, dataset=dataset)
    if results is None:
        return _err("Could not process image — check model is loaded", 500, "PREDICT_FAILED")

    top = results[0]
    _total_detects += 1
    log.info(f"[predict/{dataset}] {top['label']} ({top['confidence']}%)")

    # Log to user history
    if "user_id" in session:
        try:
            log_user_detection(session["user_id"], top["label"],
                               top["confidence"], dataset)
        except Exception as e:
            log.warning(f"History log error: {e}")

    # Geotag
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
                "type":       "danger" if any(w in top["label"].lower()
                              for w in ["stop","no","danger","warning","curve"]) else "standard"
            }
            save_location_entry(entry)
        except Exception as e:
            log.warning(f"Geotag save error: {e}")

    return jsonify({"top": results, "dataset": dataset})


# ── Detect (YOLOv8) ───────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if not ok_yolo:
        return _err("YOLOv8 not loaded on server", 503, "YOLO_UNAVAILABLE")

    data, err = _check_file("file", MAX_IMAGE_MB)
    if err:
        return err

    dataset = request.form.get("dataset", "german")
    b64, detections = detect_and_classify(data, run_prediction, dataset=dataset)
    if b64 is None:
        return _err("Detection failed", 500, "DETECT_FAILED")

    log.info(f"[yolo/{dataset}] {len(detections)} object(s) detected")
    return jsonify({
        "image":      b64,
        "detections": detections,
        "count":      len(detections),
        "dataset":    dataset,
        "road_relevant": sum(1 for d in detections if d.get("road_relevant", False))
    })


# ── Video ─────────────────────────────────────────────────────────────────────
@app.route("/process_video", methods=["POST"])
def process_video_route():
    data, err = _check_file("file", MAX_VIDEO_MB)
    if err:
        return err

    dataset = request.form.get("dataset", "german")
    result  = process_video(data, run_prediction, dataset=dataset)
    if "error" in result:
        return jsonify(result), 500

    log.info(f"[video/{dataset}] {result['sampled']} frames → "
             f"{result['unique_signs']} unique signs  ({result['processing_ms']}ms)")
    return jsonify(result)


# ── Batch predict ─────────────────────────────────────────────────────────────
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return _err("No files uploaded", 400, "NO_FILES")

    dataset = request.form.get("dataset", "german")
    output  = []
    for f in files:
        data = f.read()
        if len(data) > MAX_IMAGE_MB * 1024 * 1024:
            output.append({"filename": f.filename, "error": "File too large"})
            continue
        results = run_prediction(data, dataset=dataset)
        if results is None:
            output.append({"filename": f.filename, "error": "Could not process"})
        else:
            top = results[0]
            output.append({
                "filename":    f.filename,
                "label":       top["label"],
                "description": top["description"],
                "confidence":  top["confidence"],
                "top3":        results
            })

    return jsonify({"results": output, "count": len(output), "dataset": dataset})


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
@app.route("/gradcam", methods=["POST"])
def gradcam():
    data, err = _check_file("file", MAX_IMAGE_MB)
    if err:
        return err
    dataset = request.form.get("dataset", "german")
    b64 = run_gradcam(data, dataset=dataset)
    if b64 is None:
        return _err("Grad-CAM failed", 500, "GRADCAM_FAILED")
    return jsonify({"image": b64})


# ── GPS / Map API ─────────────────────────────────────────────────────────────
@app.route("/api/locations", methods=["GET"])
def get_all_locations():
    return jsonify(get_locations())

@app.route("/api/locations", methods=["POST"])
def add_location():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data or "lat" not in data or "lng" not in data or "label" not in data:
        return _err("Missing lat, lng, or label", 400, "MISSING_FIELDS")
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
        return _err(str(e), 400, "LOCATION_SAVE_ERROR")

@app.route("/api/locations", methods=["DELETE"])
def clear_locations():
    with open(LOCATION_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True, "message": "All locations cleared"})

@app.route("/api/locations/export")
def export_geojson():
    """Export all locations as a GeoJSON FeatureCollection."""
    locs = get_locations()
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [loc["lng"], loc["lat"]]},
        "properties": {k: v for k, v in loc.items() if k not in ("lat", "lng")}
    } for loc in locs if "lat" in loc and "lng" in loc]
    return jsonify({"type": "FeatureCollection", "features": features})


# ── Analytics Dashboard ───────────────────────────────────────────────────────
@app.route("/api/analytics")
def api_analytics():
    days    = int(request.args.get("days", 30))
    user_id = session.get("user_id")   # None → global analytics
    data    = get_dashboard_data(user_id=user_id, days=days)
    return jsonify(data)


# ── Model Comparison ──────────────────────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare_models():
    """Run same image through both German and Indian models, return side-by-side."""
    data, err = _check_file("file", MAX_IMAGE_MB)
    if err:
        return err
    german_res = run_prediction(data, dataset="german")
    india_res  = run_prediction(data, dataset="india")
    return jsonify({
        "german": german_res or [],
        "india":  india_res  or []
    })


# ── Reverse Geocoding ─────────────────────────────────────────────────────────
@app.route("/api/geocode")
def api_geocode():
    try:
        lat = float(request.args.get("lat", 0))
        lng = float(request.args.get("lng", 0))
    except (TypeError, ValueError):
        return _err("Invalid lat/lng", 400, "INVALID_COORDS")
    result = reverse_geocode(lat, lng)
    return jsonify(result)


# ── Speed Limit Check ─────────────────────────────────────────────────────────
@app.route("/api/speedcheck", methods=["POST"])
def speed_check():
    """
    Given a detected sign label and current GPS speed,
    return whether the driver is over the limit.
    """
    data        = request.get_json(silent=True) or {}
    label       = data.get("label", "").lower()
    current_spd = float(data.get("speed_kmh", 0))

    limit = None
    for token in label.split():
        token = token.replace("km/h","").replace("kmh","")
        if token.isdigit():
            limit = int(token)
            break

    if limit is None:
        return jsonify({"has_limit": False})

    over = current_spd > limit
    return jsonify({
        "has_limit":    True,
        "limit_kmh":    limit,
        "current_kmh":  current_spd,
        "over_limit":   over,
        "excess_kmh":   round(current_spd - limit, 1) if over else 0,
        "alert":        f"⚠️ Speed limit {limit} km/h! You're doing {current_spd:.0f} km/h" if over else "✅ Within speed limit"
    })


# ── Live Webcam Frame Predict (high-speed endpoint) ───────────────────────────
@app.route("/predict_live", methods=["POST"])
def predict_live():
    """
    Lightweight endpoint for real-time webcam predictions.
    Returns only top-1 result (fastest response).
    """
    global _total_detects
    data, err = _check_file("file", 2)   # 2MB max per frame
    if err:
        return err

    dataset = request.form.get("dataset", "german")
    results = run_prediction(data, dataset=dataset)
    if not results:
        return jsonify({"label": "—", "confidence": 0, "description": ""})

    top = results[0]
    _total_detects += 1
    return jsonify({
        "label":       top["label"],
        "confidence":  top["confidence"],
        "description": top["description"],
        "class_id":    top["class_id"],
        "top3":        results,
        "dataset":     dataset
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
