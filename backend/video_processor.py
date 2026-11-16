"""
video_processor.py
Extracts frames from an uploaded video, runs CNN on each frame,
returns a summary of detected traffic signs with timestamps.
"""

import cv2
import numpy as np
import base64
import os
import tempfile


def process_video(video_bytes, predict_fn, dataset="german",
                  sample_every_sec=0.5, max_frames=60):
    """
    1. Save video bytes to temp file
    2. Extract one frame every `sample_every_sec` seconds
    3. Run CNN predict on each frame
    4. Return summary + thumbnail grid (base64)

    Returns: dict with keys:
        - detections: list of {time, label, confidence, description}
        - summary:    dict of {label: count}
        - thumbnail:  base64 JPEG of collage (top detections)
        - fps, total_frames, duration
    """
    # ── Write video to temp file ──────────────────────────────────────────────
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    detections = []
    frames_data = []   # (time_sec, frame_img, top_pred)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration    = total_frames / fps
        interval    = max(1, int(fps * sample_every_sec))

        frame_idx = 0
        sampled   = 0

        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                # Encode frame to bytes for CNN
                _, buf      = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buf.tobytes()

                results = predict_fn(frame_bytes, dataset=dataset)
                if results:
                    top  = results[0]
                    time_sec = frame_idx / fps
                    detections.append({
                        "time":        round(time_sec, 1),
                        "time_str":    _fmt_time(time_sec),
                        "label":       top["label"],
                        "confidence":  top["confidence"],
                        "description": top.get("description", ""),
                        "class_id":    top.get("class_id", -1)
                    })
                    frames_data.append((time_sec, frame, top))

                sampled += 1

            frame_idx += 1

        cap.release()

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Build summary: count per label ───────────────────────────────────────
    summary = {}
    for d in detections:
        lbl = d["label"]
        summary[lbl] = summary.get(lbl, 0) + 1

    # ── Build top-6 thumbnail collage ─────────────────────────────────────────
    thumbnail_b64 = _make_collage(frames_data[:6])

    return {
        "detections":   detections,
        "summary":      summary,
        "thumbnail":    thumbnail_b64,
        "fps":          round(fps, 1),
        "total_frames": total_frames,
        "duration":     round(duration, 1),
        "sampled":      len(detections),
        "dataset":      dataset
    }


def _fmt_time(sec):
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def _make_collage(frames_data, thumb_w=200, thumb_h=150, cols=3):
    """Make a collage of up to 6 frame thumbnails with labels."""
    if not frames_data:
        return None

    rows  = (len(frames_data) + cols - 1) // cols
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    for idx, (time_sec, frame, top) in enumerate(frames_data):
        r = idx // cols
        c = idx % cols

        thumb = cv2.resize(frame, (thumb_w, thumb_h))

        # Draw label bar
        label_text = f"{top['label']} {top['confidence']}%"
        cv2.rectangle(thumb, (0, thumb_h - 24), (thumb_w, thumb_h),
                      (0, 160, 60), -1)
        cv2.putText(thumb, label_text, (4, thumb_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Draw timestamp
        cv2.putText(thumb, _fmt_time(time_sec), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        y1, y2 = r * thumb_h, (r + 1) * thumb_h
        x1, x2 = c * thumb_w, (c + 1) * thumb_w
        canvas[y1:y2, x1:x2] = thumb

    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")
