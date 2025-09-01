from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="C:/Users/a13pi/Downloads/Project_one_basketball/ShotTracker/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    name="rim_custom_final"
)
