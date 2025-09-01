from ultralytics import YOLO
from pathlib import Path

project_root = Path(__file__).resolve().parent
rim_yaml = project_root / "data.yaml"   # points to the file above

if not rim_yaml.exists():
    raise FileNotFoundError(f"Rim data.yaml not found: {rim_yaml}")

model = YOLO("yolov8n.pt")
model.train(
    data=str(rim_yaml),
    epochs=200,
    imgsz=640,
    batch=16,
    name="rim_custom_final",                    # <-- matches video_reader.py
    project=str(project_root / "runs" / "detect")
)
