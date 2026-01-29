import os
import time
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, Response
from ultralytics import YOLO
from PIL import Image
import base64
import io
face_detection_bp = Blueprint('face_detection', __name__)

_routes_dir = os.path.dirname(__file__)

# Dlib logic removed for stability on Render deployment
DLIB_AVAILABLE = False
_detector = None
_predictor = None

_project_root = os.path.abspath(os.path.join(_routes_dir, '..', '..'))
model_path = os.path.join(_project_root, 'best.pt')

if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    # Fallback if best.pt is missing (rare in this flow)
    model = YOLO('yolov8n.pt')

# Names for class filtering
_names = model.names if hasattr(model, 'names') else {}
_face_class_ids = {i for i, n in _names.items() if isinstance(n, str) and 'face' in n.lower()}
if not _face_class_ids:
    _face_class_ids = {i for i, n in _names.items() if isinstance(n, str) and n.lower() == 'person'}


def _decode_image_from_request(file_storage):
    """Decode an uploaded file into an RGB numpy array."""
    image = Image.open(file_storage.stream).convert('RGB')
    return np.array(image)


def _encode_image_to_base64_bgr(image_rgb):
    """Encode an RGB image to base64 after converting to BGR for JPEG encoding."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr)
    return base64.b64encode(buffer).decode('utf-8')


@face_detection_bp.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Read and process the image
        image_array = _decode_image_from_request(file)

        # Run detection with stricter thresholds to reduce false positives
        # First pass – face-only model, moderate confidence
        results = model(image_array, conf=0.25, iou=0.6, verbose=False)

        # If nothing is found, optionally fall back to a generic model (COCO classes)
        if (not results or results[0].boxes is None or len(results[0].boxes) == 0):
            generic_path = os.path.join(_project_root, 'src', 'yolov8n.pt')
            if os.path.exists(generic_path) and generic_path != _model_path:
                generic_model = YOLO(generic_path)
                results = generic_model(image_array, conf=0.25, iou=0.6, verbose=False)
 
        # Process results
        detections = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = _names.get(class_id, str(class_id))

                    # Only keep face/person classes if known
                    if _face_class_ids and class_id not in _face_class_ids:
                        continue

                    # Filter out extremely small boxes that are unlikely to be valid faces
                    if (x2 - x1) * (y2 - y1) < 10 * 10:
                        continue

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': class_name
                    })

        # Draw bounding boxes on the image
        annotated_image = results[0].plot() if results else image_array

        # Convert to base64 for sending to frontend
        img_base64 = _encode_image_to_base64_bgr(annotated_image)

        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'processed_image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@face_detection_bp.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Process a single frame from live camera feed"""
    try:
        # Main
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'No frame data received'}), 400

        # Decode base64 frame
        frame_data = data['frame'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        frame_bytes = base64.b64decode(frame_data)

        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame for faster inference (maintain fixed size for YOLO)
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Run detection with thresholds
        t0 = time.time()
        results = model(frame_resized, conf=0.25, iou=0.6, verbose=False)

        # Fallback to generic model if nothing is detected - This enables OBJECT detection
        if (not results or results[0].boxes is None or len(results[0].boxes) == 0):
             # Load generic model for object detection if specific model fails
             try:
                 generic_model = YOLO('yolov8n.pt')
                 results = generic_model(frame_resized, conf=0.25, iou=0.6, verbose=False)
                 mode_text = "Mode: Object Detection (Fallback)"
             except:
                 mode_text = "Mode: Face Detection"
        else:
             mode_text = "Mode: Face Detection (Best)"
        
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0

        # Process results
        detections = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = _names.get(class_id, str(class_id))

                    # Filter removed to allow ALL objects (Face + Object detection)
                    # if _face_class_ids and class_id not in _face_class_ids:
                    #     continue

                    # Minimum box area filter
                    if (x2 - x1) * (y2 - y1) < 10 * 10:
                        continue

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': class_name
                    })

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot() if results else frame_resized

        # Convert annotated_frame to grayscale for dlib
        # Dlib landmark detection removed for stability


        # Add FPS and Mode info to the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Convert back to base64
        frame_base64 = _encode_image_to_base64_bgr(annotated_frame)

        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'fps': fps,
            'mode': mode_text,
            'processed_frame': f'data:image/jpeg;base64,{frame_base64}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

