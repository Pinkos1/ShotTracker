from ultralytics import YOLO
model = YOLO("yolov8n.pt")


model.train(
    data="C:/Users/a13pi/Downloads/Project_one_basketball/ShotTracker/Detect basketball.v1i.yolov8/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    name="basketball_custom_final",   
)
