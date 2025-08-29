import torch
from ultralytics import YOLO
import yaml
import tempfile
import os

if __name__ == '__main__':
    print("Using CPU for training since no NVIDIA GPU is available.")

    # Dataset configuration
    dataset_config = {
        'path': r'C:\Users\Hp\Desktop\AI pt3\pt2_dataset_50_2025-08-29',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 41,
        'names': ['apple', 'backpack', 'banana', 'bench', 'bicycle', 'book', 'bottle', 'bowl', 'broccoli', 'cake', 'car', 'carrot', 'cell phone', 'chair', 'clock', 'couch', 'cup', 'dining table', 'dog', 'fork', 'handbag', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange', 'oven', 'person', 'pizza', 'potted plant', 'refrigerator', 'sandwich', 'sink', 'spoon', 'suitcase', 'teddy bear', 'tv', 'umbrella', 'vase', 'wine glass'],
    }

    # Create a temporary YAML file for YOLO
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_yaml:
        yaml.dump(dataset_config, temp_yaml)
        yaml_path = temp_yaml.name

    print(f"Temporary YAML file created at: {yaml_path}")

    try:
        # Use a smaller model for CPU
        model = YOLO('yolo11s.pt').to('cpu')
        print("Model device:", model.device)
        print("Starting training on CPU...")

        # Train the model
        results = model.train(
            data=yaml_path,
            epochs=50,       # Reduced epochs for CPU training
            imgsz=640,       # Smaller image size
            batch=1,         # Keep batch size small for CPU
            device='cpu'     # Force CPU
        )

    finally:
        os.remove(yaml_path)
        print(f"Temporary YAML file {yaml_path} deleted.")
