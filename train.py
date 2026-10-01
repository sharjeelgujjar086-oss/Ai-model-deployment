from ultralytics import YOLO

# Start from a pretrained YOLO checkpoint.
# Replace this filename with the checkpoint you choose to use.
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)

# After training, copy:
# runs/detect/train/weights/best.pt
# into this repository as best.pt
