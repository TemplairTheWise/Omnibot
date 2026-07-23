import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

# --- Configuration ---
onnx_model_path = "best.onnx"
calibration_dir = "train/images" 
output_hef_name = "yolo26_beverage_containers.hef"

# 1. Initialize for Raspberry Pi AI HAT
runner = ClientRunner(hw_arch="hailo8l")

# 2. Parse the ONNX model to Hailo Archive (HAR) format
print("Parsing ONNX model...")
runner.translate_onnx_model(
    onnx_model_path,
    "yolo26_beverages",
    start_node_names=["images"],
    end_node_names=["/model.23/Transpose"] # Using the safe end-node we found
)

# 3. Create a Calibration Dataset Array (Fixed: Returning a NumPy Array)
def get_calib_data():
    valid_images = [f for f in os.listdir(calibration_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    batch = []
    for img_name in valid_images[:128]: # 128 images is optimal for calibration
        img_path = os.path.join(calibration_dir, img_name)
        img = Image.open(img_path).convert('RGB').resize((640, 640))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Ultralytics ONNX expects NCHW format (Channels, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))
        batch.append(img_array)
    
    # Return a single massive array of shape (128, 3, 640, 640)
    return np.array(batch, dtype=np.float32)

# 4. Optimize (Quantize) the model
print("Loading images into memory...")
calib_data = get_calib_data()

print("Optimizing and Quantizing...")
runner.optimize(calib_data) # Pass the NumPy array directly!

# 5. Compile to HEF
print("Compiling to HEF format...")
hef = runner.compile()

with open(output_hef_name, "wb") as f:
    f.write(hef)

print(f"Success! Model compiled and saved to {output_hef_name}")
