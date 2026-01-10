import cv2
import numpy as np

# ============ CONFIG ============
MASK_PATH = "croped_mask.png"
VIDEO_PATH = "croped_video.mp4"
MIN_AREA = 1000  # noise filter
# ================================


def decode_parking_mask(mask_path):
    mask = cv2.imread(mask_path, 0)

    # 1. Convert to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 2. Clean noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)

    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area > MIN_AREA:
            boxes.append((x, y, w, h))

    # 4. Sort top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes, mask


def overlay_mask(frame, mask, alpha=0.4):
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 1] = 0  # remove green
    mask_colored[:, :, 0] = 0  # remove blue

    blended = cv2.addWeighted(frame, 1, mask_colored, alpha, 0)
    return blended


def draw_boxes(frame, boxes):
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
        cv2.putText(frame, str(i+1), (x+5, y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame


# ============ MAIN ============
boxes, mask = decode_parking_mask(MASK_PATH)
print(f"Detected slots: {len(boxes)}")

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Parking", cv2.WINDOW_NORMAL)

ret, frame = cap.read()
if not ret:
    print("Video not loading")
    exit()

h, w, _ = frame.shape
mask = cv2.resize(mask, (w, h))  # only once

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay mask
    blended = overlay_mask(frame, mask)

    # Draw boxes
    blended = draw_boxes(blended, boxes)

    cv2.imshow("Parking", blended)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
