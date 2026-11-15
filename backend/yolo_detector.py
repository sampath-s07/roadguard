"""
yolo_detector.py
YOLOv8 traffic sign detector.
Detects regions in image → crops each → classifies with CNN.
"""

import cv2
import numpy as np
import base64
import os

# Lazy-load ultralytics so app starts even if not installed
_yolo_model = None
_yolo_ok    = False

def load_yolo():
    global _yolo_model, _yolo_ok
    try:
        from ultralytics import YOLO
        # yolov8n.pt downloads automatically on first run (~6 MB)
        _yolo_model = YOLO("yolov8n.pt")
        _yolo_ok = True
        print("✅ YOLOv8n loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ YOLOv8 not available: {e}")
        _yolo_ok = False
        return False


def detect_and_classify(img_bytes, predict_fn, dataset="german"):
    """
    1. Run YOLOv8 on full image → get bounding boxes
    2. For each box → crop → run CNN classifier
    3. Return annotated image (base64) + list of detections

    predict_fn: function(img_bytes, dataset) → top3 list
    """
    if not _yolo_ok:
        return None, []

    # Decode image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []

    orig = img.copy()
    h, w = img.shape[:2]

    # ── Run YOLOv8 detection ───────────────────────────────────
    results = _yolo_model(img, verbose=False)[0]
    boxes   = results.boxes

    detections = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_conf       = float(box.conf[0])
            yolo_cls        = int(box.cls[0])
            yolo_label      = _yolo_model.names[yolo_cls]

            # Clamp box to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # ── Crop and classify with CNN ──
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            _, crop_buf = cv2.imencode(".jpg", crop)
            crop_bytes  = crop_buf.tobytes()
            cnn_results = predict_fn(crop_bytes, dataset=dataset)

            if cnn_results:
                top      = cnn_results[0]
                sign_lbl = top["label"]
                sign_pct = top["confidence"]
            else:
                sign_lbl = yolo_label
                sign_pct = round(yolo_conf * 100, 1)

            detections.append({
                "box":         [x1, y1, x2, y2],
                "yolo_class":  yolo_label,
                "yolo_conf":   round(yolo_conf * 100, 1),
                "sign_label":  sign_lbl,
                "confidence":  sign_pct,
                "top3":        cnn_results or []
            })

            # ── Draw bounding box ──
            color = (0, 200, 80)   # green
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{sign_lbl} {sign_pct}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by = max(y1 - 4, th + 4)
            cv2.rectangle(orig, (x1, by - th - 4), (x1 + tw + 6, by + 2), color, -1)
            cv2.putText(orig, label_text,
                        (x1 + 3, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    else:
        # No detections — draw "No signs detected" text
        cv2.putText(orig, "No objects detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Encode annotated image to base64
    _, buf = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64    = base64.b64encode(buf).decode("utf-8")
    return b64, detections
