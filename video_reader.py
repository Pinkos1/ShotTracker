##
# \file video_reader.py
# \brief Plays uploaded video and detects basketballs (COCO) and rims (custom) using YOLOv8.
# \author Adam Pinkos
# \date 5/7/25

import cv2
from ultralytics import YOLO

# Load models
ball_model = YOLO("yolov8n.pt")
rim_model = YOLO("runs/detect/rim_custom_final/weights/best.pt")  #

##
# \brief Reads and plays a video at a specified speed, drawing detections for basketballs and rims
# \param filepath The full path to the video file.
# \param speed_factor Playback speed multiplier
def read_video(filepath, speed_factor = 20):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Try to get FPS to adjust playback delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / (fps * speed_factor)) if fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # increase brightness
        bright_frame = cv2.convertScaleAbs(frame, alpha = 1.2, beta = 40)

        # detects basketball
        ball_results = ball_model(bright_frame, conf = 0.25)
        for r in ball_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 32:  # class 32 = sports ball
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(bright_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bright_frame, "Basketball", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Basketball:", x1, y1, x2, y2)

        # detects rim
        rim_results = rim_model(bright_frame, conf = 0.5)
        for r in rim_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(bright_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(bright_frame, "Rim", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print("Rim:", x1, y1, x2, y2)

        # frames
        cv2.imshow("Basketball + Rim Detection", bright_frame)

        if cv2.waitKey(delay) & 0xFF == 27:  # 
            break

    cap.release()
    cv2.destroyAllWindows()
