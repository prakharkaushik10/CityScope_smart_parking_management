# 🚗 Automatic Number Plate Recognition (ANPR)
### ESP32-CAM + YOLOv5 + PaddleOCR (CUDA 11.8)

This project implements a **real-time Automatic Number Plate Recognition (ANPR)** system using:

- 📷 **ESP32-CAM** for live video streaming
- 🧠 **YOLOv5 (custom trained)** for license plate detection
- 🔍 **PaddleOCR (GPU)** for plate text recognition
- ⚡ **NVIDIA GPU (CUDA 11.8)** for acceleration

---

## ✨ Features

- Real-time number plate detection
- Indian license plate optimized
- ESP32-CAM live stream support
- GPU-accelerated detection & OCR
- Robust OCR post-processing
- Handles mirrored / inverted ESP32 images

---

## 📁 Project Structure

ANPR/\
├── anpr.py # Main ANPR script\
├── test.py # Local testing script\
├── best.pt # Trained YOLOv5 model\
├── requirements.txt # Dependencies\
├── .env # Environment variables\
├── .gitignore\
├── README.md\
└── yolov5/ # YOLOv5 repo (optional)

yaml
Copy code

---

## 🧪 Requirements

- **Python 3.10**
- **NVIDIA GPU**
- **CUDA 11.8**
- Windows / Linux

---

## ⚙️ Installation (Recommended Order)

```bash
python -m venv env
env\Scripts\activate

pip install --upgrade pip

pip install numpy==1.26.4
pip install opencv-python==4.6.0.66 pillow

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install paddlepaddle-gpu==2.6.1
pip install paddleocr==2.7.0.3

pip install -r requirements.txt
