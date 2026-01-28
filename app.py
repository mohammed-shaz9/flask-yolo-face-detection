import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import dlib

# Set page configuration
st.set_page_config(page_title="YOLOv8 Face Detection", layout="wide")

st.title("🚀 YOLOv8 Face Detection & Landmarks")
st.write("Upload an image to detect faces and view 68-point landmarks.")

# --- Load Models ---
@st.cache_resource
def load_models():
    # Paths for YOLO
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_project_root, 'runs', 'detect', 'train_epochs202', 'weights', 'best.pt'),
        os.path.join(_project_root, 'runs', 'detect', 'train7', 'weights', 'best.pt'),
        os.path.join(_project_root, 'src', 'yolov8n.pt'),
        os.path.join(_project_root, 'yolov8n.pt'),
    ]
    _model_path = next((p for p in _candidates if os.path.exists(p)), 'yolov8n.pt')
    model = YOLO(_model_path)
    
    # Path for Dlib
    _predictor_path = os.path.join(_project_root, 'src', 'routes', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(_predictor_path):
         # Fallback search
         for root, dirs, files in os.walk(_project_root):
             if 'shape_predictor_68_face_landmarks.dat' in files:
                 _predictor_path = os.path.join(root, 'shape_predictor_68_face_landmarks.dat')
                 break
                 
    predictor = dlib.shape_predictor(_predictor_path)
    return model, predictor

try:
    with st.spinner("Loading AI models..."):
        model, predictor = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.6)

# --- Main App ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert('RGB')
    image_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Detecting faces..."):
        # Run YOLO Inference
        results = model(image_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Process Results
        annotated_image = results[0].plot() # YOLO annotations
        
        # Add Dlib Landmarks
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        boxes = results[0].boxes
        
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Apply dlib landmarks within the box
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                try:
                    landmarks = predictor(gray_image, dlib_rect)
                    for i in range(0, landmarks.num_parts):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
                except:
                    continue

    with col2:
        st.subheader("Processed Image")
        st.image(annotated_image, use_container_width=True)
        
    st.write(f"Detected **{len(results[0].boxes)}** potential faces.")
