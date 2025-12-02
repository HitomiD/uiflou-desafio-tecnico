import torch
from ultralytics import YOLO

def load(model_path: str) -> YOLO:
    print(f"Model path: {model_path}")
    model = YOLO(model_path)

    if torch.cuda.is_available():
        print("CUDA available: Switching to GPU...")
        model.to("cuda")
    
    return model