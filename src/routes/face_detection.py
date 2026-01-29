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
        
        # Check if ONNX model exists, if not, try to export it
        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found. Attempting to export from PyTorch model...")
            if os.path.exists(pt_path):
                try:
                    # Lazy import to avoid memory overhead if not needed
                    from ultralytics import YOLO
                    logger.info("Loading YOLO to export...")
                    model = YOLO(pt_path)
                    logger.info("Exporting to ONNX...")
                    model.export(format='onnx', imgsz=416)
                    logger.info("Export complete!")
                    
                    # Force cleanup
                    del model
                    import gc
                    import torch
                    gc.collect()
                except Exception as e:
                    logger.error(f"Failed to export model: {e}")
            else:
                logger.error("Neither ONNX nor PyTorch model found!")

        logger.info(f"Looking for ONNX model at: {onnx_path}")
        
        if not os.path.exists(onnx_path):
             logger.error("best.onnx could not be loaded!")
             raise FileNotFoundError("best.onnx missing and export failed")

        # Load ONNX model
        # Use CPU provider for Render compatibility
        providers = ['CPUExecutionProvider']
        _session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output metadata
        _input_name = _session.get_inputs()[0].name
        _output_names = [output.name for output in _session.get_outputs()]
        
        logger.info(f"ONNX model loaded! Input: {_input_name}")
        return _session
        
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def preprocess(image_array):
    """Preprocess image for YOLOv8 ONNX (Resize, Normalize, Transpose)."""
    # Resize to 416x416 (matching export)
    input_shape = (416, 416)
    img_h, img_w = image_array.shape[:2]
    
    img = cv2.resize(image_array, input_shape)
    
    # Normalize (0-255 -> 0.0-1.0)
    img = img.astype(np.float32) / 255.0
    
    # HWC -> CHW (3, 416, 416)
    img = img.transpose((2, 0, 1))
    
    # Add batch dim (1, 3, 416, 416)
    img = np.expand_dims(img, axis=0)
    
    return img, img_w, img_h

def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    # boxes: [x1, y1, x2, y2]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=iou_threshold)
    return indices

def postprocess(outputs, img_w, img_h):
    """Parse YOLOv8 ONNX output."""
    # Output shape is typically [1, 5, 3549] (for 1 class + 4 coords)
    # 4 coords (cx, cy, w, h) + 1 class score (face)
    
    output = outputs[0][0]  # Remove batch dim: [5, 3549]
    
    # Transpose to [3549, 5]
    output = output.transpose()
    
    boxes = []
    scores = []
    class_ids = []
    
    # Iterate through predictions
    # Format: [cx, cy, w, h, score]
    for row in output:
        score = row[4]
        if score < 0.25: # Confidence threshold
            continue
            
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        
        # Convert to [x1, y1, x2, y2]
        x1 = (cx - w/2)
        y1 = (cy - h/2)
        x2 = (cx + w/2)
        y2 = (cy + h/2)
        
        # Scale back to original image size
        # Model input was 416x416
        x_scale = img_w / 416
        y_scale = img_h / 416
        
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
        
        boxes.append([x1, y1, x2-x1, y2-y1]) # cv2 NMS needs [x, y, w, h]
        scores.append(float(score))
        class_ids.append(0) # Logic assumes 1 class (face)
        
    # Apply NMS
    indices = nms(boxes, scores)
    
    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            final_detections.append({
                'bbox': [x, y, x+w, y+h], # x1, y1, x2, y2
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
        session = get_session()
        return jsonify({
            'status': 'healthy',
            'model': 'ONNX (Lightweight)',
            'input': _input_name
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@face_detection_bp.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        
        image_array = _decode_image_from_request(file)
        
        # Preprocess
        input_tensor, w, h = preprocess(image_array)
        
        # Inference
        session = get_session()
        outputs = session.run(_output_names, {_input_name: input_tensor})
        
        # Postprocess
        detections = postprocess(outputs, w, h)
        
        # Draw boxes
        annotated_image = image_array.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        img_base64 = _encode_image_to_base64_bgr(annotated_image)
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
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for speed if needed (ONNX is fast though)
        # Using native resolution for now, or 416x416 consistent with model
        t0 = time.time()
        
        input_tensor, w, h = preprocess(frame_rgb)
        session = get_session()
        outputs = session.run(_output_names, {_input_name: input_tensor})
        detections = postprocess(outputs, w, h)
        
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        
        # Draw
        annotated_frame = frame_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add stats
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Mode: ONNX (Ultra-Light)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        frame_base64 = _encode_image_to_base64_bgr(annotated_frame)
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
