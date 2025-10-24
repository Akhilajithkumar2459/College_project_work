from ultralytics import YOLO

# --- MODIFICATIONS START ---

# 1. Load the custom model architecture for small object detection.
#    Replace 'path/to/your/repo/models/yolov8-p2.yaml' with the actual path.
print("Loading custom model architecture from yolov8-p2.yaml...")
model = YOLO('/home/hutlab_int/Akhil_yolo/Research/yolov8/Yolov8-Small-Object-Detection-Arial-Images/cfg/model/yolov8-p2.yaml')

# 2. Load pre-trained weights into the custom architecture (Transfer Learning).
#    This starts training from a knowledgeable base, not from scratch.
#    'yolov8s.pt' is a good starting point for a small model.
print("Loading pre-trained weights from yolov8s.pt...")
model=YOLO('yolov8s')

# --- MODIFICATIONS END ---

# 3. Train the model on your dataset.
#    This part remains the same, as it correctly points to your data.
print("Starting training...")
results = model.train(data="/home/hutlab_int/Akhil_yolo/Research/yolov8/data.yaml",
                      epochs=150,
                      imgsz=640,
                      batch=32,
                      device=0) # Using GPU 0

print("Training finished successfully.")