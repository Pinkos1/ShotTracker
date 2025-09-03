# \file video_reader.py
# \brief Plays uploaded video and (separately) provides a fast scan mode for detections.
# \author Adam Pinkos
# \date 5/7/25

import time
import math
import cv2
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------
# LOAD MODELS ONCE (smallest viable; switch to your custom weights)
# -------------------------------------------------------------------
# If you have custom weights, keep these paths; else start with yolov8n for speed.
BALL_WEIGHTS = "runs/detect/basketball_custom_final/weights/best.pt"
RIM_WEIGHTS  = "runs/detect/rim_custom_final/weights/best.pt"

# Fallback to tiny COCO model if custom missing (optional)
def _safe_yolo(path, fallback="yolov8n.pt"):
    try:
        return YOLO(path)
    except Exception:
        return YOLO(fallback)

ball_model = _safe_yolo(BALL_WEIGHTS)
rim_model  = _safe_yolo(RIM_WEIGHTS)

# Try to use GPU + half precision for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
for m in (ball_model, rim_model):
    m.to(DEVICE)
    try:
        m.fuse()  # small speedup
    except Exception:
        pass

USE_HALF = DEVICE == "cuda"
if USE_HALF:
    try:
        ball_model.model.half()
        rim_model.model.half()
    except Exception:
        USE_HALF = False  # some backends/layers may not support half

# -------------------------------------------------------------------
# RUNTIME: PLAYBACK (your original)
# -------------------------------------------------------------------
def read_video(filepath, speed_factor=20):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / (fps * speed_factor)) if fps and fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # brighten
        bright_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)

        # BALLS
        ball_results = ball_model(bright_frame, conf=0.5, imgsz=640, verbose=False)
        for r in ball_results:
            for box in r.boxes:
                cls = int(box.cls[0]) if box.cls is not None else -1
                # if your custom dataset has basketball as class 0; otherwise remove this check
                if cls in (-1, 0):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(bright_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bright_frame, "Basketball", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Basketball:", x1, y1, x2, y2)

        # RIMS
        rim_results = rim_model(bright_frame, conf=0.5, imgsz=640, verbose=False)
        for r in rim_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(bright_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(bright_frame, "Rim", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print("Rim:", x1, y1, x2, y2)

        cv2.imshow("Basketball + Rim Detection", bright_frame)
        if cv2.waitKey(delay) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------
# FAST SCAN: no GUI, frame skipping, resize, half precision
# Goal: cover the video quickly (~target_seconds).
# Returns summary that you can extend for make/miss logic.
# -------------------------------------------------------------------
def scan_video(filepath, target_seconds=30, min_stride=1, max_stride=20, base_imgsz=640):
    """
    Quickly scan a video for basketball & rim detections.

    Strategy:
    - Downscale to imgsz (default 640, try 480 for even faster).
    - Auto-tune frame stride after a short warm-up so we finish ~target_seconds.
    - No imshow or waits; pure compute.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {"error": "Cannot open video."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Quick warm-up on ~30 frames to estimate throughput
    warm_n = min(30, max(5, total_frames // 300)) or 10
    warm_stride = 5  # sample sparsely
    _ = _run_n_frames(cap, warm_n, warm_stride, base_imgsz)

    # Estimate per-frame cost by a small timed batch
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t0 = time.time()
    measured = _run_n_frames(cap, warm_n, 1, base_imgsz)  # contiguous for stable timing
    t1 = time.time()
    per_frame_sec = (t1 - t0) / max(1, measured["frames_analyzed"])

    # Choose a stride to hit the target time budget
    # total_inferences ≈ total_frames / stride; total_time ≈ per_frame_sec * total_inferences
    # stride ≈ (per_frame_sec * total_frames) / target_seconds
    est_stride = int(math.ceil((per_frame_sec * max(1, total_frames)) / max(1, target_seconds)))
    frame_stride = max(min_stride, min(max_stride, est_stride if est_stride > 0 else min_stride))

    # Full pass with chosen stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start = time.time()
    ball_count = 0
    rim_count  = 0
    frames_analyzed = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip frames quickly using CAP_PROP_POS_FRAMES (fast seek on many codecs)
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        frames_analyzed += 1

        # Downscale for speed (preserve aspect)
        frame_small = _resize_short_side(frame, base_imgsz)

        # Inference (silent)
        ball_res = ball_model(frame_small, conf=0.5, imgsz=base_imgsz, verbose=False)
        rim_res  = rim_model(frame_small,  conf=0.5, imgsz=base_imgsz, verbose=False)

        # Count detections (you can refine with class IDs as needed)
        ball_count += sum(len(r.boxes) for r in ball_res)
        rim_count  += sum(len(r.boxes) for r in rim_res)

        frame_idx += 1

    elapsed = time.time() - start
    cap.release()
    return {
        "frames_analyzed": frames_analyzed,
        "frame_stride": frame_stride,
        "ball_count": int(ball_count),
        "rim_count": int(rim_count),
        "elapsed_sec": float(elapsed),
        "video_frames": total_frames,
        "fps": float(fps),
        "device": DEVICE,
        "half": USE_HALF,
        "imgsz": base_imgsz
    }

# Helpers
def _resize_short_side(img, short_side):
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    if h < w:
        new_h = short_side
        new_w = int(w * (short_side / h))
    else:
        new_w = short_side
        new_h = int(h * (short_side / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def _run_n_frames(cap, n, stride, imgsz):
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    frames = 0
    got = 0
    # read until collected n frames that meet the stride
    while got < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        pos = start_pos + frames - 1
        if pos % stride != 0:
            continue
        # quick downscale+inference with ball model only to estimate speed
        frame_small = _resize_short_side(frame, imgsz)
        _ = ball_model(frame_small, conf=0.4, imgsz=imgsz, verbose=False)
        got += 1
    return {"frames_read": frames, "frames_analyzed": got}
