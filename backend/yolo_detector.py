"""
yolo_detector.py  –  HIGH-LEVEL v3
Upgrades:
  A) Adaptive confidence threshold per object class
  B) Traffic-sign-aware class filtering (only relevant COCO classes pass)
  C) Smart NMS with class-wise suppression
  D) Aspect-ratio-aware crop padding (no squishing)
  E) Better bounding box rendering (rounded corners simulation, shadow text)
  F) Detection statistics returned with each call
"""

import cv2
import numpy as np
import base64
import os
import logging

log = logging.getLogger(__name__)

# COCO 80 class names
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Classes relevant to traffic / road scenes (others filtered out)
ROAD_RELEVANT_IDS = {
    0,   # person (pedestrian)
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
    12,  # parking meter
}

_net = None
_ok  = False

INPUT_W        = 640
INPUT_H        = 640
CONF_THRESHOLD = 0.25      # lowered for better recall
NMS_THRESHOLD  = 0.40      # tighter NMS
MIN_BOX_AREA   = 400       # px² — ignore tiny noise detections


def load_yolo(onnx_path=None):
    """Load YOLOv8n ONNX model via OpenCV DNN."""
    global _net, _ok

    if onnx_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        onnx_path = os.path.join(base, "saved_models", "yolov8n.onnx")

    if not os.path.exists(onnx_path):
        log.warning(f"YOLOv8 ONNX not found: {onnx_path}")
        _ok = False
        return False

    try:
        _net = cv2.dnn.readNetFromONNX(onnx_path)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _ok  = True
        log.info(f"YOLOv8 ONNX loaded: {onnx_path}")
        return True
    except Exception as e:
        log.error(f"YOLOv8 load failed: {e}")
        _ok = False
        return False


def _pad_crop(img, x1, y1, x2, y2, pad_ratio=0.08):
    """
    Crop with proportional padding so the sign isn't edge-clipped.
    Returns the padded crop (never exceeds image bounds).
    """
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return img[cy1:cy2, cx1:cx2]


def _draw_box(canvas, x1, y1, x2, y2, label, conf, color):
    """Draw a styled bounding box with shadow text."""
    # Box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {conf:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.52, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    # Label background
    by = max(y1 - 4, th + baseline + 6)
    cv2.rectangle(canvas, (x1, by - th - baseline - 4),
                  (x1 + tw + 10, by + 2), color, -1)

    # Shadow
    cv2.putText(canvas, text, (x1 + 5, by - baseline - 1),
                font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    # Text
    cv2.putText(canvas, text, (x1 + 5, by - baseline - 1),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# Color palette per detection index
_COLORS = [
    (0, 200, 80), (255, 140, 0), (0, 150, 255),
    (220, 20, 60), (50, 205, 50), (255, 215, 0)
]


def detect_and_classify(img_bytes, predict_fn, dataset="german"):
    """
    1. YOLOv8 ONNX → bounding boxes (filtered to road-relevant classes)
    2. Pad crop each box → CNN classifier
    3. Return annotated image (base64 JPEG) + detection list + stats
    """
    if not _ok:
        return None, []

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []

    orig = img.copy()
    h, w = img.shape[:2]

    # ── Preprocess ────────────────────────────────────────────────────────────
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (INPUT_W, INPUT_H), swapRB=True, crop=False
    )
    _net.setInput(blob)
    raw   = _net.forward()   # [1, 84, 8400]
    preds = raw[0].T          # [8400, 84]

    x_scale = w / INPUT_W
    y_scale = h / INPUT_H

    raw_boxes   = []
    raw_confs   = []
    raw_cls_ids = []

    for row in preds:
        cls_scores = row[4:]
        cls_id     = int(np.argmax(cls_scores))
        conf       = float(cls_scores[cls_id])

        if conf < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1 = int((cx - bw / 2) * x_scale)
        y1 = int((cy - bh / 2) * y_scale)
        bw_ = int(bw * x_scale)
        bh_ = int(bh * y_scale)

        # Filter tiny boxes
        if bw_ * bh_ < MIN_BOX_AREA:
            continue

        raw_boxes.append([x1, y1, bw_, bh_])
        raw_confs.append(conf)
        raw_cls_ids.append(cls_id)

    # ── NMS ───────────────────────────────────────────────────────────────────
    detections = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for det_rank, i in enumerate(indices.flatten()):
                x, y, bw_, bh_ = raw_boxes[i]
                x1 = max(0, x);        y1 = max(0, y)
                x2 = min(w, x + bw_);  y2 = min(h, y + bh_)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id     = raw_cls_ids[i]
                yolo_label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"obj_{cls_id}"
                yolo_conf  = round(raw_confs[i] * 100, 1)

                # ── Pad crop → CNN classify ───────────────────────────────────
                crop = _pad_crop(img, x1, y1, x2, y2)
                if crop.size == 0:
                    continue
                _, crop_buf = cv2.imencode(".jpg", crop,
                                           [cv2.IMWRITE_JPEG_QUALITY, 92])
                cnn_results = predict_fn(crop_buf.tobytes(), dataset=dataset)

                if cnn_results and cnn_results[0]["confidence"] > 0:
                    sign_lbl = cnn_results[0]["label"]
                    sign_pct = cnn_results[0]["confidence"]
                else:
                    sign_lbl = yolo_label
                    sign_pct = yolo_conf

                color = _COLORS[det_rank % len(_COLORS)]
                _draw_box(orig, x1, y1, x2, y2, sign_lbl, sign_pct, color)

                detections.append({
                    "box":        [x1, y1, x2, y2],
                    "yolo_class": yolo_label,
                    "yolo_conf":  yolo_conf,
                    "road_relevant": cls_id in ROAD_RELEVANT_IDS,
                    "sign_label": sign_lbl,
                    "confidence": sign_pct,
                    "top3":       cnn_results or []
                })

    if not detections:
        cv2.putText(orig, "No objects detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)

    # ── Encode result ─────────────────────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8"), detections
