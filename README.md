---
title: Yolo Face Detection
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# YOLOv8 Face Detection Project

**Quick-Start (for Reviewers/Examiners)**

1. Install Python ≥ 3.10 on your machine.
2. From the project root, create & activate a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   # source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Launch the web app:
   ```bash
   ./venv/Scripts/python.exe src/main.py   # Windows
   # OR simply: python src/main.py
   ```
4. Open your browser to **http://127.0.0.1:5000** and start uploading images or use the live-camera tab.  
   (No additional configuration is needed; the default port is 5000.)

_The remainder of this README gives a more detailed walkthrough, background, and troubleshooting tips._

---

This project implements a basic face detection web application using a pre-trained YOLOv8 model. It's designed to be simple to set up and run, making it suitable for beginners in deep learning and web development.

## Project Description

This application detects and localizes faces (or more generally, people, as the base YOLOv8n model is trained on the 'person' class) in images uploaded by the user. The results are displayed with bounding boxes and confidence scores on a web interface.

## Features

- Web-based interface for easy interaction.
- Image upload via drag-and-drop or file selection.
- Real-time inference using a pre-trained YOLOv8n model.
- Visual display of detected objects with bounding boxes.
- Details of detections (class, confidence, bounding box coordinates).

## Setup Instructions (for Beginners)

Follow these steps to get the project up and running on your local machine:

### 1. Clone the Repository (or extract the ZIP file)

If you received a ZIP file, extract it to your desired location. If this were a GitHub repository, you would clone it:

```bash
git clone <repository_url>
cd yolov8-face-detection
```

### 2. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Download the Pre-trained YOLOv8n Model

The `yolov8n.pt` model is a general object detection model. For face detection, it will primarily detect 'person' objects. If you want a model specifically trained on faces, you would need to fine-tune this model on a face dataset or find a pre-trained YOLOv8-Face model.

Download the `yolov8n.pt` model and place it in the `src/` directory:

```bash
python -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\')" # This command downloads the model
# The model will be saved in ~/.cache/ultralytics/ultralytics/yolov8n.pt
# You need to manually copy it to the src/ directory of this project
cp ~/.cache/ultralytics/ultralytics/yolov8n.pt src/yolov8n.pt
```

**Note:** The `yolov8n.pt` model detects general objects, including 'person'. For specific 'face' detection, fine-tuning on a face dataset (like WIDER FACE) would be required, or using a model pre-trained specifically for faces.

### 5. Run the Flask Application

Navigate to the project root directory and run the Flask application:

```bash
cd src
python main.py
```

### 6. Access the Web Application

Open your web browser and go to `http://127.0.0.1:5000` (or `http://localhost:5000`).

## Project Structure

```
. (project root)
├── venv/                 # Python virtual environment
├── src/
│   ├── main.py           # Main Flask application entry point
│   ├── yolov8n.pt        # Pre-trained YOLOv8n model
│   ├── routes/
│   │   └── face_detection.py # Flask blueprint for face detection API
│   └── static/
│       └── index.html    # Frontend HTML, CSS, and JavaScript
└── requirements.txt      # Python dependencies
```

## Key Concepts for Beginners

- **YOLOv8 (You Only Look Once):** A state-of-the-art real-time object detection model. It's known for its speed and accuracy.
- **Object Detection:** The task of identifying and locating objects within an image or video. It draws bounding boxes around detected objects and assigns a class label and confidence score.
- **Pre-trained Models:** Models that have already been trained on a very large dataset (like COCO). This saves a lot of training time and computational resources. You can use them directly for inference or fine-tune them for specific tasks.
- **Flask:** A lightweight Python web framework used to build web applications and APIs.
- **HTML, CSS, JavaScript:** The foundational technologies for building web pages. HTML structures the content, CSS styles it, and JavaScript adds interactivity.
- **API (Application Programming Interface):** A set of rules that allows different software applications to communicate with each other. In this project, the web frontend communicates with the Flask backend via an API.

## Customization and Further Steps

- **Fine-tuning for Faces:** To make the model specifically detect 'faces' instead of 'persons', you would need to fine-tune the `yolov8n.pt` model on a dedicated face dataset (e.g., WIDER FACE, CelebA). This involves training the model for additional epochs on your specific dataset.
- **Facial Landmark Detection:** This project focuses on bounding box detection. Integrating facial landmark detection would involve using a separate model or a specialized YOLOv8 variant (like YOLOv8-Face) that outputs keypoints on the face.
- **Deployment:** For permanent deployment, you would typically use cloud platforms (AWS, Google Cloud, Azure) and production-grade web servers (Gunicorn, Nginx) instead of the Flask development server.

Feel free to explore and modify the code to deepen your understanding!

