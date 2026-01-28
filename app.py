import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# Set page configuration
st.set_page_config(page_title="YOLOv8 AI Detection", layout="wide")

st.title("🚀 YOLOv8 Live AI Detection")
st.write("Detect faces and objects in real-time.")

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
                 
    predictor = None
    if DLIB_AVAILABLE and os.path.exists(_predictor_path):
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
detect_mode = st.sidebar.radio("Detection Mode", ["Face Detection", "All Objects"])

# --- Video Processor ---
if HAS_WEBRTC:
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.predictor = predictor
            self.conf = conf_threshold
            self.iou = iou_threshold
            self.mode = detect_mode

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Update params
            self.conf = conf_threshold
            self.iou = iou_threshold
            self.mode = detect_mode

            # Run YOLO
            results = self.model(img, conf=self.conf, iou=self.iou, verbose=False)
            annotated_img = results[0].plot()

            # Add Landmarks if in Face Mode
            if self.mode == "Face Detection" and DLIB_AVAILABLE and self.predictor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                    try:
                        landmarks = self.predictor(gray, dlib_rect)
                        for i in range(landmarks.num_parts):
                            p = landmarks.part(i)
                            cv2.circle(annotated_img, (p.x, p.y), 2, (0, 255, 0), -1)
                    except:
                        continue
            
            return annotated_img

# --- Main App ---
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📸 Take Photo", "🎥 Live Stream"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Processed")
            # YOLO
            results = model(image_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()
            
            # Landmarks
            if detect_mode == "Face Detection" and DLIB_AVAILABLE and predictor:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                    try:
                        landmarks = predictor(gray, dlib_rect)
                        for i in range(landmarks.num_parts):
                            p = landmarks.part(i)
                            cv2.circle(annotated, (p.x, p.y), 2, (0, 255, 0), -1)
                    except:
                        continue
            st.image(annotated, use_container_width=True)

with tab2:
    camera_file = st.camera_input("Quick Photo Detection")
    if camera_file:
        image = Image.open(camera_file).convert('RGB')
        image_array = np.array(image)
        results = model(image_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
        annotated = results[0].plot()
        if detect_mode == "Face Detection" and DLIB_AVAILABLE and predictor:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                try:
                    landmarks = predictor(gray, dlib_rect)
                    for i in range(landmarks.num_parts):
                        p = landmarks.part(i)
                        cv2.circle(annotated, (p.x, p.y), 2, (0, 255, 0), -1)
                except:
                    continue
        st.image(annotated, use_container_width=True)

with tab3:
    if HAS_WEBRTC:
        st.subheader("Real-Time AI Feed")
        st.write("Click 'Start' to open your webcam for live detection.")
        webrtc_streamer(key="live-detection", video_transformer_factory=VideoProcessor)
    else:
        st.error("Live Stream library (streamlit-webrtc) not installed. Please install it to use this feature.")
