import os
import json
from ultralytics import YOLO

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'data.yaml')

# Prefer the longest training's best weights if available
CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_epochs202', 'weights', 'best.pt'),
    os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train7', 'weights', 'best.pt'),
    os.path.join(PROJECT_ROOT, 'src', 'yolov8n.pt'),
]
MODEL_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[-1])

OUT_DIR = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'val_run_1')
os.makedirs(OUT_DIR, exist_ok=True)
METRICS_JSON = os.path.join(OUT_DIR, 'metrics.json')


def main():
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Using data: {DATA_PATH}")

    model = YOLO(MODEL_PATH)

    # Validate
    results = model.val(
        data=DATA_PATH,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        save_json=True,
        project=os.path.join(PROJECT_ROOT, 'runs', 'detect'),
        name='val_run_1',
        exist_ok=True,
        verbose=False,
    )

    # Collect metrics
    try:
        metrics = {
            'map50_95': float(getattr(results.box, 'map', None) or 0.0),
            'map50': float(getattr(results.box, 'map50', None) or 0.0),
            'precision': float(getattr(results.box, 'mp', None) or 0.0),
            'recall': float(getattr(results.box, 'mr', None) or 0.0),
            'nc': len(getattr(model, 'names', {}) or {}),
            'model_path': MODEL_PATH,
            'data_path': DATA_PATH,
        }
    except Exception:
        # Fallback if API changes
        metrics = {'error': 'Failed to parse metrics from results'}

    # Save metrics
    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print("Validation metrics saved to:", METRICS_JSON)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()