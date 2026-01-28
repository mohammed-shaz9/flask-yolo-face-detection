---
title: Yolo Face Detection
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# YOLOv8 AI Detection Project

**Modern Face & Object Detection with Live Streaming**

## Quick-Start (Local)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the App**:
   ```bash
   streamlit run app.py
   ```
3. **Access**: Open your browser to **http://localhost:8501**.

---

## Features

- **🎥 Live Video Stream**: Real-time detection directly from your webcam using WebRTC.
- **📁 Image Upload**: Drag-and-drop any image to analyze it.
- **📸 Quick Photo**: Capture a single frame from your camera for instant processing.
- **🔍 Dual Detection Modes**:
  - **Face Detection**: Specialized mode for faces with **68-point facial landmarks** (green dots).
  - **All Objects**: Detects 80 different categories (people, cars, animals, etc.) using YOLOv8.
- **⚙️ Adjustable Settings**: Change confidence and IOU thresholds live from the sidebar.

---

## Technical Stack

- **Frontend/Backend**: [Streamlit](https://streamlit.io/)
- **AI Model**: [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics)
- **Face Landmarks**: [Dlib](http://dlib.net/) (68-point predictor)
- **Live Stream**: [Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc)
- **Deployment**: Dockerized for Hugging Face Spaces.

---

## Setup & Configuration

### Prerequisites
- Python 3.10 or 3.11.
- Webcam (for live features).

### Project Structure
```
.
├── app.py                # Main Streamlit application
├── Dockerfile            # Container configuration for Cloud
├── requirements.txt      # Python dependencies
├── yolov8n.pt           # YOLOv8 model weights
├── src/
│   └── routes/
│       └── shape_predictor_68_face_landmarks.dat # Dlib model
└── dataset/              # Original training data (optional)
```

## Concepts for Beginners

- **YOLOv8**: A "You Only Look Once" model that scans the entire image in one pass, making it incredibly fast.
- **Facial Landmarks**: These are 68 specific points on a human face (eyes, nose, mouth, jawline) used for alignment and expression analysis.
- **WebRTC**: A technology that allows high-quality video streaming in the browser without plugins.

---

*Developed for Face Detection and AI Object Recognition research.*
