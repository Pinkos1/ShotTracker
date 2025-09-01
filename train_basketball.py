from ultralytics import YOLO
from pathlib import Path

# Base folder = ShotTracker
project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "Detect basketball.v1i.yolov8 (1)"   # match exact folder name
data_yaml = dataset_dir / "data.yaml"

if not data_yaml.exists():
    raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

print(f"Using dataset YAML: {data_yaml}")

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_yaml),
    epochs=200,
    imgsz=640,
    batch=16,
    name="basketball_custom_final",
    project=str(project_root / "runs" / "detect")
)
