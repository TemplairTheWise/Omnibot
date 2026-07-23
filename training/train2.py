from ultralytics import YOLO

# Load the pretrained YOLO26 Nano model
model = YOLO("yolo26n.pt")

# Train the model on your custom beverage dataset using your GTX 1080
results = model.train(
    data=f"/home/templair/Downloads/Beverage_Containers_Model/data.yaml", 
    epochs=100, 
    imgsz=640,
    device=0  # This explicitly assigns the training task to your GTX 1080
)
