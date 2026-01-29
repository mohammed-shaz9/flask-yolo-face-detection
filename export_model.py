from ultralytics import YOLO
import os

# Load model
model = YOLO('best.pt')

# Export to ONNX
# This creates 'best.onnx' in the same directory
path = model.export(format='onnx', imgsz=416, optimize=True)
print(f"Model exported to {path}")
