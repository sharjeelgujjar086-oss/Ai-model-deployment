import cv2
import re
import easyocr
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.seen_plates = {}

    def reset(self):
        self.seen_plates.clear()

    @staticmethod
    def clean_text(text):
        text = text.upper()
        return re.sub(r"[^A-Z0-9-]", "", text)

    def read_plate(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return ""

        plate_image = cv2.resize(
            plate_image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        results = self.reader.readtext(plate_image)
        if not results:
            return ""

        best = max(results, key=lambda x: x[2])
        return self.clean_text(best[1])

    def process_frame(self, frame):
        output = []

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            h, w = frame.shape[:2]

            for i in range(len(boxes)):
                x1, y1, x2, y2 = (
                    boxes.xyxy[i].cpu().numpy().astype(int)
                )
                confidence = float(boxes.conf[i].cpu().item())

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().item())

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                text = self.read_plate(crop)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )

                label = text or "plate"
                if track_id is not None:
                    label = f"{label} ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                item = {
                    "plate": text,
                    "confidence": round(confidence, 3),
                    "track_id": track_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                }

                if text:
                    self.seen_plates[text] = item
                    output.append(item)

        return frame, output
