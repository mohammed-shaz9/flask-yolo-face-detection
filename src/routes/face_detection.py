import os
import time
import cv2
import numpy as np
import logging
from flask import Blueprint, request, jsonify
from PIL import Image
import base64
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_detection_bp = Blueprint('face_detection', __name__)

_routes_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_routes_dir, '..', '..'))

# Global model variable
_session = None
_input_name = None
_output_names = None

def get_session():
    """Lazy load ONNX Runtime session with auto-export."""
    global _session, _input_name, _output_names
    
    if _session is not None:
        return _session
    
    try:
        import onnxruntime as ort
        
        onnx_path = os.path.join(_project_root, 'best.onnx')
        pt_path = os.path.join(_project_root, 'best.pt')
        
        # Check if ONNX model exists
        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found. Attempting to export from PyTorch model...")
            if os.path.exists(pt_path):
                from ultralytics import YOLO
                model = YOLO(pt_path)
                model.export(format='onnx', imgsz=416)
                del model
                gc.collect()
            else:
                logger.error("No model found!")

        # Load ONNX model with CPU provider
        providers = ['CPUExecutionProvider']
        _session = ort.InferenceSession(onnx_path, providers=providers)
        
        _input_name = _session.get_inputs()[0].name
        _output_names = [output.name for output in _session.get_outputs()]
        
        logger.info(f"ONNX model loaded! Input: {_input_name}")
        return _session
        
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        raise

def preprocess(image_array):
    """Preprocess image for YOLOv8 (Resize to 416x416, Normalize, Transpose)."""
    img_h, img_w = image_array.shape[:2]
    # YOLOv8 usually expects 416x416 or 640x640
    input_shape = (416, 416)
    img = cv2.resize(image_array, input_shape)
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0) # Add batch dim
    return img, img_w, img_h

def postprocess(outputs, img_w, img_h):
    """Parse YOLOv8 ONNX output correctly."""
    # Output shape should be [1, 4 + num_classes, 3549]
    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0] # Remove batch dim
    
    # Transpose so rows are detections: [3549, 5]
    output = output.transpose()
    
    boxes = []
    scores = []
    
    # YOLOv8 format: [cx, cy, w, h, class0, class1...]
    # Since we have only 1 class ('face'), output is 0,1,2,3 for box and 4 for score
    for row in output:
        score = row[4]
        if score > 0.15: # Lowered threshold for better sensitivity
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # Map back to original image size
            # Model was set to 416x416
            x_scale = img_w / 416
            y_scale = img_h / 416
            
            # Convert center to top-left
            x1 = int((cx - w/2) * x_scale)
            y1 = int((cy - h/2) * y_scale)
            bw = int(w * x_scale)
            bh = int(h * y_scale)
            
            boxes.append([x1, y1, bw, bh])
            scores.append(float(score))
            
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.15, nms_threshold=0.45)
    
    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x1, y1, bw, bh = box
            final_detections.append({
                'bbox': [max(0, x1), max(0, y1), min(img_w, x1+bw), min(img_h, y1+bh)],
                'confidence': scores[i],
                'class': 'face'
            })
            
    return final_detections

def _decode_image_from_request(file_storage):
    image = Image.open(file_storage.stream).convert('RGB')
    return np.array(image)

def _encode_image_to_base64_bgr(image_rgb):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr)
    return base64.b64encode(buffer).decode('utf-8')

@face_detection_bp.route('/health', methods=['GET'])
def health_check():
    try:
        get_session()
        return jsonify({'status': 'healthy', 'model': 'ONNX Face Detector'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@face_detection_bp.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        
        image_array = _decode_image_from_request(file)
        input_tensor, w, h = preprocess(image_array)
        
        session = get_session()
        outputs = session.run(_output_names, {_input_name: input_tensor})
        detections = postprocess(outputs, w, h)
        
        # Draw results
        annotated_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Draw green bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            label = f"Face: {det['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        img_base64 = base64.b64encode(cv2.imencode('.jpg', annotated_image)[1]).decode('utf-8')
        gc.collect()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'processed_image': f'data:image/jpeg;base64,{img_base64}'
        })
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@face_detection_bp.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        t0 = time.time()
        input_tensor, w, h = preprocess(frame_rgb)
        session = get_session()
        outputs = session.run(_output_names, {_input_name: input_tensor})
        detections = postprocess(outputs, w, h)
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        
        # Draw on frame
        # We work on frame_bgr for drawing and returning
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Green bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label
            label = f"Face {det['confidence']:.2f}"
            cv2.putText(frame_bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Stats info
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Faces: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        frame_base64 = base64.b64encode(cv2.imencode('.jpg', frame_bgr)[1]).decode('utf-8')
        gc.collect()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'fps': fps,
            'processed_frame': f'data:image/jpeg;base64,{frame_base64}'
        })
    except Exception as e:
        logger.error(f"Frame error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
