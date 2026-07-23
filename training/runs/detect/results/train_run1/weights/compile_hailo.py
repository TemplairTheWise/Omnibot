import os
import numpy as np
import tensorflow as tf
from PIL import Image
from hailo_sdk_client import ClientRunner

model_name = "yolo26n-beverage-1.0"
onnx_path = "best.onnx"
hef_path = f"{model_name}.hef"
calib_path = "/home/templair/Downloads/Beverage_Containers_Model/valid/images"

def calibration_loader():
    images = [f for f in os.listdir(calib_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        raise FileNotFoundError(f"No images found in {calib_path}")

    subset_images = images[:100]
    print(f"Loading {len(subset_images)} images for calibration...")
    
    data_list = []
    for filename in subset_images:
        filepath = os.path.join(calib_path, filename)
        
        # Load, Resize, Normalize
        img = Image.open(filepath).convert('RGB')
        img = img.resize((640, 640))
        data = np.array(img).astype(np.float32) / 255.0
        
        data_list.append(data)
    
    # Convert to Numpy Array: Shape (100, 640, 640, 3)
    np_data = np.array(data_list)
    
    # 1. Create TF Dataset
    # This splits the array into individual items.
    # Each item 'x' has shape (640, 640, 3)
    ds = tf.data.Dataset.from_tensor_slices(np_data)
    
    # 2. VITAL FIX: Map to (x, 0)
    # - Tuple structure fixes 'TensorSpec' error.
    # - Index 0 is just 'x' (640, 640, 3). This fixes 'BadInputsShape'.
    # - The '0' acts as a dummy label in case the SDK tries to unpack (x, y).
    ds = ds.map(lambda x: (x, 0))
    
    # 3. NO BATCHING
    # We intentionally skip .batch() to keep the shape (640, 640, 3).
    
    return ds

# --- Main Compilation Flow ---
print(f"Parsing {onnx_path}...")
runner = ClientRunner(hw_arch="hailo8l")

end_nodes = [
    '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv',
    '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv',
    '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv',
    '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv',
    '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv',
    '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv'
]

runner.translate_onnx_model(
    onnx_path, 
    model_name,
    end_node_names=end_nodes
)

print("Running Optimization...")
calib_dataset = calibration_loader()
runner.optimize(calib_dataset)

print("Compiling to HEF...")
hef = runner.compile()

with open(hef_path, "wb") as f:
    f.write(hef)

print(f"Success! Saved to {hef_path}")
