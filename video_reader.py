## 
# \file video_reader.py
# \brief Plays uploaded video and detects basketballs using YOLOv8.
# \author Adam Pinkos
# \date 5/7/25

import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")  # or "yolov8s.pt" for better accuracy

##
# \brief Reads and plays a video at a specified speed, drawing detections for basketballs.
# \param filepath The full path to the video file.
# \param speed_factor Playback speed multiplier (e.g. 5 for 5x speed).
def read_video(filepath, speed_factor=5):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / (fps * speed_factor)) if fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO object detection on the frame
        results = model(frame)

        # Draw detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 37:  # class 37 = sports ball (could be basketball)
                    xyxy = box.xyxy[0].tolist()
                    cv2.rectangle(frame,
                                  (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "Basketball", (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Basketball Detection", frame)
        if cv2.waitKey(delay) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
