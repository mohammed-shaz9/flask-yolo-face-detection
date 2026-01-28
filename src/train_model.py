from ultralytics import YOLO
import torch

def train_model():
    # Load the model
    model = YOLO('yolov8n.pt')  # load pretrained model

    # Training arguments with optimized parameters
    args = {
        'data': 'C:\\Users\\SHAZ\\Downloads\\face_detection_project\\home\\ubuntu\\face_detection_project_zip\\dataset\\data.yaml',        # path to data file
        'epochs': 100,                      # number of epochs
        'patience': 20,                     # early stopping patience
        'batch': 16,                        # batch size
        'imgsz': 640,                       # input image size
        'save': True,                       # save checkpoints
        'cache': True,                      # cache images for faster training
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # use GPU if available
        'workers': 8,                       # number of worker threads
        'optimizer': 'Adam',                # optimizer
        'lr0': 0.001,                      # initial learning rate
        'lrf': 0.01,                       # final learning rate
        'momentum': 0.937,                 # SGD momentum/Adam beta1
        'weight_decay': 0.0005,            # optimizer weight decay
        'warmup_epochs': 3,                # warmup epochs
        'warmup_momentum': 0.8,            # warmup initial momentum
        'warmup_bias_lr': 0.1,             # warmup initial bias lr
        'augment': True,                   # use data augmentation
        'name': 'fine_tuned_face',         # save to project/name
        'exist_ok': True,                  # existing project/name ok
        'pretrained': True,                # use pretrained model
        'dropout': 0.2,                    # use dropout for better generalization
    }

    # Start training
    try:
        results = model.train(**args)
        print("Training completed successfully!")
        return results
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return None

if __name__ == '__main__':
    train_model()
