import subprocess
import numpy as np
import cv2
import time
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ================= CONFIG =================
PI_IP = os.getenv("PI_IP")
PORT = int(os.getenv("SRT_PORT", "9000"))

WIDTH = 640
HEIGHT = 480
FPS = 25
# ==========================================

if not PI_IP:
    raise RuntimeError("PI_IP not set in .env file")

print(f"Connecting to SRT stream at {PI_IP}:{PORT} ... Press Q to quit")

ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "quiet",
    "-fflags", "+genpts",
    "-analyzeduration", "1000000",
    "-probesize", "1000000",
    "-i", f"srt://{PI_IP}:{PORT}?mode=caller&latency=120",
    "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
    "-pix_fmt", "bgr24",
    "-f", "rawvideo",
    "-"
]

pipe = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=0
)

frame_size = WIDTH * HEIGHT * 3
buffer = b""
last_frame_time = time.time()

while True:
    chunk = pipe.stdout.read(frame_size - len(buffer))
    if not chunk:
        if time.time() - last_frame_time < 5:
            continue
        print("Stream ended")
        break

    buffer += chunk

    if len(buffer) < frame_size:
        continue

    raw = buffer[:frame_size]
    buffer = buffer[frame_size:]
    last_frame_time = time.time()

    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

    cv2.imshow("SRT Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipe.terminate()
cv2.destroyAllWindows()
print("Client exited cleanly")
