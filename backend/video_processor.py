"""
video_processor.py  –  HIGH-LEVEL v3
Upgrades:
  A) Adaptive frame sampling (scene-change detection → only extract changed frames)
  B) Confidence-filtered detections (only high-confidence signs logged)
  C) Deduplication: same label within 2 sec window is merged
  D) Top-N sign timeline with per-sign peak confidence
  E) Better collage: larger thumbnails, sign category color coding
  F) Returns processing_time_ms for performance monitoring
"""

import cv2
import numpy as np
import base64
import os
import tempfile
import time
import logging

log = logging.getLogger(__name__)

CONF_MIN         = 30.0    # only log detections above this %
DEDUP_WINDOW_SEC = 2.0     # merge same label within this window
MAX_FRAMES       = 80      # max frames to sample
SCENE_DIFF_THOLD = 25.0    # mean pixel diff to trigger scene-change sample


def process_video(video_bytes, predict_fn, dataset="german",
                  sample_every_sec=0.5, max_frames=MAX_FRAMES):
    """
    1. Write video to temp file
    2. Scene-change aware frame sampling
    3. CNN predict on sampled frames (high-confidence only)
    4. Deduplicate repeat detections within 2s window
    5. Return summary + thumbnail collage + timing

    Returns dict with:
        detections, summary, thumbnail, fps, total_frames, duration,
        sampled, unique_signs, dataset, processing_ms
    """
    t_start = time.time()

    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    detections  = []
    frames_data = []   # (time_sec, frame_bgr, top_pred)
    prev_gray   = None

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps
        interval     = max(1, int(fps * sample_every_sec))

        frame_idx = 0
        sampled   = 0
        last_det_time = {}   # label → last logged time (for dedup)

        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                # Scene-change detection: skip if frame looks same as previous
                gray = cv2.cvtColor(
                    cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY
                ).astype("float32")

                if prev_gray is not None:
                    diff = float(np.mean(np.abs(gray - prev_gray)))
                    if diff < SCENE_DIFF_THOLD and sampled > 0:
                        frame_idx += 1
                        continue   # skip similar frame

                prev_gray = gray

                _, buf      = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                results     = predict_fn(buf.tobytes(), dataset=dataset)
                time_sec    = frame_idx / fps

                if results:
                    top  = results[0]
                    conf = top["confidence"]

                    # Confidence filter
                    if conf >= CONF_MIN:
                        label = top["label"]

                        # Deduplication
                        last_t = last_det_time.get(label, -999)
                        if (time_sec - last_t) >= DEDUP_WINDOW_SEC:
                            last_det_time[label] = time_sec
                            detections.append({
                                "time":        round(time_sec, 1),
                                "time_str":    _fmt_time(time_sec),
                                "label":       label,
                                "confidence":  conf,
                                "description": top.get("description", ""),
                                "class_id":    top.get("class_id", -1)
                            })
                            frames_data.append((time_sec, frame.copy(), top))

                sampled += 1

            frame_idx += 1

        cap.release()

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Summary: label → {count, peak_confidence} ────────────────────────────
    summary = {}
    for d in detections:
        lbl = d["label"]
        if lbl not in summary:
            summary[lbl] = {"count": 0, "peak_confidence": 0.0}
        summary[lbl]["count"] += 1
        summary[lbl]["peak_confidence"] = max(
            summary[lbl]["peak_confidence"], d["confidence"]
        )

    # Sort by count desc for display
    summary_sorted = dict(
        sorted(summary.items(), key=lambda kv: kv[1]["count"], reverse=True)
    )

    # Collage: top 6 by confidence
    top_frames = sorted(frames_data,
                        key=lambda x: x[2]["confidence"], reverse=True)[:6]
    thumbnail_b64 = _make_collage(top_frames)

    processing_ms = round((time.time() - t_start) * 1000)
    log.info(f"[video] {sampled} sampled, {len(detections)} detections, "
             f"{len(summary)} unique signs, {processing_ms}ms")

    return {
        "detections":    detections,
        "summary":       summary_sorted,
        "thumbnail":     thumbnail_b64,
        "fps":           round(fps, 1),
        "total_frames":  total_frames,
        "duration":      round(duration, 1),
        "sampled":       sampled,
        "unique_signs":  len(summary),
        "dataset":       dataset,
        "processing_ms": processing_ms
    }


def _fmt_time(sec):
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


# Category → color (BGR)
_SIGN_COLORS = {
    "speed":      (0, 100, 220),
    "stop":       (0, 0, 200),
    "no":         (0, 50, 200),
    "warning":    (0, 165, 255),
    "turn":       (200, 100, 0),
    "roundabout": (150, 0, 150),
    "default":    (0, 160, 60)
}


def _sign_color(label):
    l = label.lower()
    for key, color in _SIGN_COLORS.items():
        if key in l:
            return color
    return _SIGN_COLORS["default"]


def _make_collage(frames_data, thumb_w=240, thumb_h=180, cols=3):
    """Make a styled collage of up to 6 frame thumbnails with labels."""
    if not frames_data:
        return None

    rows   = (len(frames_data) + cols - 1) // cols
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    for idx, (time_sec, frame, top) in enumerate(frames_data):
        r, c   = idx // cols, idx % cols
        thumb  = cv2.resize(frame, (thumb_w, thumb_h),
                            interpolation=cv2.INTER_AREA)
        color  = _sign_color(top["label"])
        bar_h  = 28

        # Color bar at bottom
        cv2.rectangle(thumb, (0, thumb_h - bar_h), (thumb_w, thumb_h), color, -1)

        # Label (truncated)
        lbl_txt = top["label"][:22] + ".." if len(top["label"]) > 24 else top["label"]
        cv2.putText(thumb, f"{lbl_txt}  {top['confidence']:.0f}%",
                    (4, thumb_h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Timestamp badge
        cv2.rectangle(thumb, (2, 2), (60, 20), (0, 0, 0), -1)
        cv2.putText(thumb, _fmt_time(time_sec), (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 80), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), color, 2)

        y1, y2 = r * thumb_h, (r + 1) * thumb_h
        x1, x2 = c * thumb_w, (c + 1) * thumb_w
        canvas[y1:y2, x1:x2] = thumb

    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")
