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
# \param speed_factor Playback speed multiplier
def read_video(filepath, speed_factor=20):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # original FPS to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    delay = 1
    #int(1000 / (fps * speed_factor)) if fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Increase brightness to improve visibility and detection accuracy
        bright_frame = cv2.convertScaleAbs(frame, alpha = 1.2, beta = 40)

        # increased birghtness for frams 
        results = model(bright_frame, conf=0.1)  

        # bounding boxes 
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 32: 
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(bright_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bright_frame, "Basketball", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Drawing basketball box at:", x1, y1, x2, y2)

        # Display the processed frame with detections
        cv2.imshow("Basketball", bright_frame)

        if cv2.waitKey(delay) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
