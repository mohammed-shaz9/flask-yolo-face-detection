import os
import time
import cv2
import numpy as np
import logging
from flask import Blueprint, request, jsonify
from PIL import Image
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_detection_bp = Blueprint('face_detection', __name__)

_routes_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_routes_dir, '..', '..'))

# Global model variable - lazy loaded
_model = None
_model_names = {}
_face_class_ids = set()


def get_model():
    """Lazy load the YOLO model on first use."""
    global _model, _model_names, _face_class_ids
    
    if _model is not None:
        return _model
    
    try:
        from ultralytics import YOLO
        
        model_path = os.path.join(_project_root, 'best.pt')
        model_path_fallback = os.path.join(_project_root, 'yolov8n.pt')
        
        logger.info(f"Project root: {_project_root}")
        logger.info(f"Looking for model at: {model_path}")
        logger.info(f"Model exists: {os.path.exists(model_path)}")
        logger.info(f"Fallback exists: {os.path.exists(model_path_fallback)}")
        
        # List files in project root for debugging
        try:
            files = os.listdir(_project_root)
            logger.info(f"Files in project root: {files}")
        except Exception as e:
            logger.error(f"Could not list project root: {e}")
        
        if os.path.exists(model_path):
            logger.info("Loading best.pt model...")
            _model = YOLO(model_path)
            logger.info("best.pt model loaded successfully!")
        elif os.path.exists(model_path_fallback):
            logger.info("Loading yolov8n.pt fallback model...")
            _model = YOLO(model_path_fallback)
            logger.info("yolov8n.pt model loaded successfully!")
        else:
            logger.info("No local model found, downloading yolov8n.pt...")
            _model = YOLO('yolov8n.pt')
            logger.info("yolov8n.pt downloaded and loaded!")
        
        # Set up class names
        _model_names = _model.names if hasattr(_model, 'names') else {}
        logger.info(f"Model class names: {_model_names}")
        
        _face_class_ids = {i for i, n in _model_names.items() if isinstance(n, str) and 'face' in n.lower()}
        if not _face_class_ids:
            # If no 'face' class, use 'person' class (for general object detection)
            _face_class_ids = {i for i, n in _model_names.items() if isinstance(n, str) and n.lower() == 'person'}
        logger.info(f"Face/Person class IDs: {_face_class_ids}")
        
        return _model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _decode_image_from_request(file_storage):
    """Decode an uploaded file into an RGB numpy array."""
    image = Image.open(file_storage.stream).convert('RGB')
    return np.array(image)


def _encode_image_to_base64_bgr(image_rgb):
    """Encode an RGB image to base64 after converting to BGR for JPEG encoding."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr)
    return base64.b64encode(buffer).decode('utf-8')


@face_detection_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is working."""
    try:
        model = get_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'model_classes': list(_model_names.values()) if _model_names else [],
            'project_root': _project_root,
            'files': os.listdir(_project_root) if os.path.exists(_project_root) else []
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'project_root': _project_root
        }), 500


@face_detection_bp.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Get model (lazy load)
        model = get_model()
        
        # Read and process the image
        image_array = _decode_image_from_request(file)
        logger.info(f"Image shape: {image_array.shape}")

        # Run detection
        results = model(image_array, conf=0.25, iou=0.6, verbose=False)
        logger.info(f"Detection results: {len(results[0].boxes) if results and results[0].boxes else 0} boxes")

        # Process results
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = _model_names.get(class_id, str(class_id))

                # Filter out extremely small boxes
                if (x2 - x1) * (y2 - y1) < 100:
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
        logger.error(f"Detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@face_detection_bp.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Process a single frame from live camera feed"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'No frame data received'}), 400

        # Get model (lazy load)
        model = get_model()

        # Decode base64 frame
        frame_data = data['frame'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        frame_bytes = base64.b64decode(frame_data)

        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame for faster inference
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Run detection
        t0 = time.time()
        results = model(frame_resized, conf=0.25, iou=0.6, verbose=False)
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0

        # Process results
        detections = []
        mode_text = "Mode: Face Detection"
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = _model_names.get(class_id, str(class_id))

                # Minimum box area filter
                if (x2 - x1) * (y2 - y1) < 100:
                    continue

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot() if results else frame_resized

        # Add FPS and Mode info to the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
        logger.error(f"Frame detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500
