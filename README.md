# AI License Plate Recognition API

A beginner-friendly FastAPI project for license-plate detection, OCR, and video tracking.

## Important

This repository contains the application code, but **not the trained `best.pt` weights**.

You need a license-plate detection model trained for your dataset. Put it in the repository root:

```text
plate_ai/
├── best.pt
├── main.py
├── plate_engine.py
├── train.py
└── requirements.txt
```

Only use plate data you are authorized to process, and follow applicable privacy/local laws.

## Run in VS Code

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your model

Put your trained model here:

```text
best.pt
```

### 4. Start FastAPI

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /detect-video` to upload an MP4/MOV/AVI/MKV file.

The response includes detected plate text, confidence, bounding boxes, tracking IDs, and a URL for the annotated output video.

## Training

Prepare a YOLO dataset with one class:

```text
license_plate
```

Use `data.yaml.example` as a starting point. Copy it to `data.yaml`, prepare the dataset folders, then run:

```bash
python train.py
```

After training, place:

```text
runs/detect/train/weights/best.pt
```

at the project root.

## Render deployment

Render's FastAPI configuration uses:

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

This repository also includes `render.yaml` for a reproducible configuration.

### Important Render limitation

Video processing and OCR can use significant CPU/RAM. A small/free instance may be slow or unsuitable for long/high-resolution videos. For production workloads, use an appropriately sized compute plan and consider a background job/worker architecture.

Also note that files written to a normal service filesystem should not be treated as permanent storage. For production, store uploaded/output videos in object storage or another persistent storage system.

## API

### Health

```http
GET /
GET /health
```

### Process video

```http
POST /detect-video
Content-Type: multipart/form-data
file=<video>
```

### Retrieve processed video

```http
GET /video/{filename}
```

## Architecture

```text
Video
  |
  v
OpenCV frames
  |
  v
YOLO license-plate detector
  |
  v
ByteTrack
  |
  v
Plate crop
  |
  v
EasyOCR
  |
  v
JSON results + annotated video
```
