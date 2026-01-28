---
title: Yolo Face Detection
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: src/main.py
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
   python src/main.py
   ```
3. **Access**: Open your browser to **http://localhost:7860**.

---

## Features

- **🎥 API & Web Interface**: Flask-based RESTful API with a web UI.
- **📁 Image Upload**: Simple HTML/JS interface for uploading images.
- **� Dual Detection Modes**:
  - **Face Detection**: Specialized mode for faces with **68-point facial landmarks** (green dots).
  - **All Objects**: Detects 80 different categories (people, cars, animals, etc.) using YOLOv8.
- **⚙️ Backend Processing**: Fast backend inference using YOLOv8 and Dlib.

---

## Technical Stack

- **Frontend**: HTML5, JavaScript (Vanilla)
- **Backend**: [Flask](https://flask.palletsprojects.com/) (Python)
- **AI Model**: [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics)
- **Face Landmarks**: [Dlib](http://dlib.net/) (68-point predictor)
- **Deployment**: Dockerized for Hugging Face Spaces.

---

## Setup & Configuration

### Prerequisites
- Python 3.10 or 3.11.
- CMake (for Dlib).

### Project Structure
```
.
├── src/                  # Flask Application Source
│   ├── main.py           # Entry point
│   └── routes/           # API Routes
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── best.pt               # YOLOv8 Fine-tuned Model
└── README.md             # Documentation
```

## Concepts for Beginners

- **YOLOv8**: A "You Only Look Once" model that scans the entire image in one pass, making it incredibly fast.
- **Facial Landmarks**: These are 68 specific points on a human face (eyes, nose, mouth, jawline) used for alignment and expression analysis.
- **WebRTC**: A technology that allows high-quality video streaming in the browser without plugins.

---

*Developed for Face Detection and AI Object Recognition research.*
