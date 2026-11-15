from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_loader import load_model, run_prediction, run_gradcam

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models", "custom_cnn.h5"
)

ok = load_model(MODEL)
if not ok:
    print("WARNING: model load failed.")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/gallery")
def gallery():
    return send_from_directory(app.static_folder, "gallery.html")

@app.route("/status")
def status():
    return jsonify({"status": "RoadGuard API running", "version": "2.0"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    raw     = f.read()
    results = run_prediction(raw)
    if results is None:
        return jsonify({"error": "could not process image"}), 500

    top = results[0]
    print(f"predicted: {top['label']} ({top['confidence']}%)")
    return jsonify({"top": results})


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    output = []
    for f in files:
        raw     = f.read()
        results = run_prediction(raw)
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
    return jsonify({"results": output})


@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "file" not in request.files:
        return jsonify({"error": "no image file"}), 400
    f   = request.files["file"]
    raw = f.read()
    b64 = run_gradcam(raw)
    if b64 is None:
        return jsonify({"error": "gradcam failed"}), 500
    return jsonify({"image": b64})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
