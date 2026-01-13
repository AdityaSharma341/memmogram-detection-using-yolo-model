from ultralytics import YOLO
import torch

def train_yolo():
    # 1. Load the model
    # We use 'yolov8n.pt' (Nano) which downloads automatically.
    # It is pre-trained on COCO, so it knows basic shapes.
    print("Loading model...")
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # data: path to your yaml file
    # epochs: number of times to go through the whole dataset (50 is a good start)
    # imgsz: image size (640 is standard)
    # batch: how many images per GPU cycle (Auto-adjusts or set to 16/32)
    # device: 0 means use the first NVIDIA GPU
    print("Starting training on GPU...")
    
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        batch=16,           # If you get "Out of Memory" error, reduce this to 8 or 4
        name='memmogram_train', # Name of the output folder
        patience=10         # Stop early if no improvement for 10 epochs
    )

    print("Training Complete!")

if __name__ == "__main__":
    # verification just in case
    if torch.cuda.is_available():
        train_yolo()
    else:
        print("Error: GPU not found. Aborting training to save time.")