"""
yolo_detector.py
YOLOv8 object detection using OpenCV DNN (no ultralytics needed).
Loads yolov8n.onnx → detects objects → crops → CNN classifies sign.
"""

import cv2
import numpy as np
import base64
import os

# COCO 80 class names (YOLOv8 pretrained on COCO)
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

_net  = None
_ok   = False

INPUT_W        = 640
INPUT_H        = 640
CONF_THRESHOLD = 0.30
NMS_THRESHOLD  = 0.45


def load_yolo(onnx_path=None):
    """Load YOLOv8n ONNX model via OpenCV DNN."""
    global _net, _ok

    if onnx_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        onnx_path = os.path.join(base, "saved_models", "yolov8n.onnx")

    if not os.path.exists(onnx_path):
        print(f"⚠️  YOLOv8 ONNX not found: {onnx_path}")
        print("    Export it in Colab: YOLO('yolov8n.pt').export(format='onnx')")
        _ok = False
        return False

    try:
        _net = cv2.dnn.readNetFromONNX(onnx_path)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _ok = True
        print(f"✅ YOLOv8 ONNX loaded via OpenCV DNN: {onnx_path}")
        return True
    except Exception as e:
        print(f"⚠️  YOLOv8 load failed: {e}")
        _ok = False
        return False


def detect_and_classify(img_bytes, predict_fn, dataset="german"):
    """
    1. Run YOLOv8 ONNX via OpenCV DNN → bounding boxes
    2. Crop each box → run CNN classifier
    3. Return annotated image (base64 JPEG) + detection list
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
    raw = _net.forward()          # shape: [1, 84, 8400]

    # ── Parse YOLOv8 output ───────────────────────────────────────────────────
    # YOLOv8 output: [1, 84, 8400]  → transpose → [8400, 84]
    # First 4 cols: cx, cy, bw, bh (in INPUT_W/H space)
    # Next 80 cols: per-class confidence scores
    preds = raw[0].T              # [8400, 84]

    x_scale = w / INPUT_W
    y_scale = h / INPUT_H

    raw_boxes   = []
    raw_confs   = []
    raw_cls_ids = []

    for row in preds:
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        scores  = row[4:]
        cls_id  = int(np.argmax(scores))
        conf    = float(scores[cls_id])

        if conf < CONF_THRESHOLD:
            continue

        x1 = int((cx - bw / 2) * x_scale)
        y1 = int((cy - bh / 2) * y_scale)
        bw_ = int(bw * x_scale)
        bh_ = int(bh * y_scale)

        raw_boxes.append([x1, y1, bw_, bh_])
        raw_confs.append(conf)
        raw_cls_ids.append(cls_id)

    # ── NMS ───────────────────────────────────────────────────────────────────
    detections = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw_, bh_ = raw_boxes[i]
                x1 = max(0, x);       y1 = max(0, y)
                x2 = min(w, x + bw_); y2 = min(h, y + bh_)
                if x2 <= x1 or y2 <= y1:
                    continue

                yolo_label = COCO_CLASSES[raw_cls_ids[i]] if raw_cls_ids[i] < len(COCO_CLASSES) else f"obj_{raw_cls_ids[i]}"
                yolo_conf  = round(raw_confs[i] * 100, 1)

                # ── Crop → CNN classify ───────────────────────────────────────
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                _, crop_buf = cv2.imencode(".jpg", crop)
                cnn_results = predict_fn(crop_buf.tobytes(), dataset=dataset)

                if cnn_results:
                    sign_lbl = cnn_results[0]["label"]
                    sign_pct = cnn_results[0]["confidence"]
                else:
                    sign_lbl = yolo_label
                    sign_pct = yolo_conf

                detections.append({
                    "box":        [x1, y1, x2, y2],
                    "yolo_class": yolo_label,
                    "yolo_conf":  yolo_conf,
                    "sign_label": sign_lbl,
                    "confidence": sign_pct,
                    "top3":       cnn_results or []
                })

                # ── Draw bounding box ─────────────────────────────────────────
                color = (0, 200, 80)
                cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
                label_text = f"{sign_lbl}  {sign_pct}%"
                (tw, th), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                by = max(y1 - 4, th + 6)
                cv2.rectangle(orig,
                              (x1, by - th - 4),
                              (x1 + tw + 8, by + 2), color, -1)
                cv2.putText(orig, label_text, (x1 + 4, by - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    if not detections:
        cv2.putText(orig, "No objects detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)

    # ── Encode result ─────────────────────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), detections
