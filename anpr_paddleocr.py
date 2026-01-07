import cv2
import torch
import re
import os
from paddleocr import PaddleOCR
from dotenv import load_dotenv

# =====================================
# LOAD ENV FILE
# =====================================
load_dotenv()

ESP32_STREAM = os.getenv("ESP32_STREAM_URL")
WEIGHTS = os.getenv("YOLO_WEIGHTS", "best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", 0.15))

if ESP32_STREAM is None:
    raise RuntimeError("❌ ESP32_STREAM_URL missing in .env file")

print("✅ Using stream:", ESP32_STREAM)

# =====================================
# LOAD YOLOv5
# =====================================
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=WEIGHTS,
    force_reload=False
)
model.conf = CONF_THRES
model.eval()

# =====================================
# LOAD OCR
# =====================================
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False
)

# =====================================
# IMAGE ORIENTATION FIX (ESP32)
# =====================================
def fix_orientation(img):
    img = cv2.flip(img, 1)  # mirror
    return img

# =====================================
# CLEAN PLATE TEXT
# =====================================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # remove INDIA badge text
    text = text.replace("IND", "").replace("IN", "")

    patterns = [
        r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}",  # MH12AB1234
        r"[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]?"      # 22BH6517A
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group()

    return ""

# =====================================
# OPEN ESP32 STREAM
# =====================================
cap = cv2.VideoCapture(ESP32_STREAM)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open ESP32 stream")

print("✅ ESP32-CAM stream connected")

# =====================================
# MAIN LOOP
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = fix_orientation(crop)

        ocr_result = ocr.ocr(crop, cls=False)

        raw_text = ""
        if ocr_result:
            for line in ocr_result:
                if line is None:
                    continue
                for word in line:
                    raw_text += word[1][0] + " "

        plate = clean_plate(raw_text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if plate:
            cv2.putText(
                frame,
                plate,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            print("🚘 PLATE:", plate)

    cv2.imshow("ESP32-CAM ANPR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
