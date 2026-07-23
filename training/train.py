from ultralytics import YOLO

# Load the YOLO26 model
# This will download the pretrained weights automatically
model = YOLO('yolo26n.pt') 

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,             # Full COCO usually needs 300+, but 100 is a good start
    imgsz=640,              # Image resolution
    batch=16,               # GTX 1080 friendly batch size
    device=0,               # Use your NVIDIA GPU
    workers=4,              # Number of CPU threads for data loading
    half=True,              # Use FP16 (Mixed Precision) to save VRAM
    optimizer='MuSGD',      # The new YOLO26 optimizer (or 'auto')
    project='results',  # Save folder name
    name='train_run1'       # Run name
)
