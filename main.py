import os
import uuid
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from plate_engine import PlateDetector

app = FastAPI(
    title="AI License Plate Recognition",
    version="1.0.0",
    description="YOLO license-plate detection + OCR + ByteTrack video API."
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

detector = PlateDetector("best.pt")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI License Plate Recognition",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename or "")[1].lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv"}

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Use: {', '.join(sorted(allowed))}"
        )

    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{extension}")
    output_filename = f"result_{uuid.uuid4()}.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Could not open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # mp4v is widely available with OpenCV; browser playback may vary.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Could not create output video.")

    detector.reset()
    detected_plates = {}
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            processed_frame, results = detector.process_frame(frame)

            for item in results:
                text = item["plate"]
                if text:
                    detected_plates[text] = item

            writer.write(processed_frame)
    finally:
        cap.release()
        writer.release()
        if os.path.exists(input_path):
            os.remove(input_path)

    return {
        "success": True,
        "frames_processed": frame_count,
        "plates_detected": len(detected_plates),
        "plates": list(detected_plates.values()),
        "output_video": f"/video/{output_filename}"
    }


@app.get("/video/{filename}")
def get_video(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video not found.")

    return FileResponse(path, media_type="video/mp4", filename=filename)
