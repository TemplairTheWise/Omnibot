import cv2
import numpy as np
import sys
from hailo_platform import (
    HEF, 
    VDevice, 
    InferVStreams, 
    ConfigureParams, 
    InputVStreamParams, 
    OutputVStreamParams, 
    FormatType,
    HailoStreamInterface
)

from pathlib import Path
IMAGE_PATH   = Path(__file__).parent / "assets" / "bottles.jpg"
HEF_PATH     = Path(__file__).parent / "models" / "yolo26_split.hef"
OUTPUT_IMAGE = Path(__file__).parent / "assets" / "output_detection.jpg"

print(f"Loading split YOLO 26 HEF from {HEF_PATH}...")
try:
    hef = HEF(str(HEF_PATH))
except Exception as e:
    print(f"Error loading HEF: {e}")
    sys.exit(1)

print("Connecting to Hailo NPU via PCIe...")
with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    input_info = hef.get_input_vstream_infos()[0]
    out_infos = hef.get_output_vstream_infos()

    # NPU handles normalization; send pure pixels, receive clean decimals
    in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    print("Preparing Inference Pipeline...")
    with InferVStreams(network_group, in_params, out_params) as infer_pipeline:
        
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"Error: Could not find {IMAGE_PATH}")
            sys.exit(1)
            
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        print("Injecting image into NPU...")
        with network_group.activate(network_group_params):
            infer_results = infer_pipeline.infer({input_info.name: np.expand_dims(img_rgb, axis=0)})
        
        # Extract the two output tensors
        out1 = np.squeeze(infer_results[out_infos[0].name])
        out2 = np.squeeze(infer_results[out_infos[1].name])
        
        # Ensure the arrays are oriented as (8400, Channels) instead of (Channels, 8400)
        if out1.shape[0] < out1.shape[-1]:
            out1 = out1.T
        if out2.shape[0] < out2.shape[-1]:
            out2 = out2.T
            
        # Dynamically assign boxes (4 channels) and logits (9 channels)
        if out1.shape[1] == 4:
            boxes = out1
            logits = out2
        else:
            boxes = out2
            logits = out1
            
        # Apply Sigmoid to turn raw logits into 0.0 to 1.0 percentages
        # (Using np.clip to prevent overflow warnings from extreme logits)
        probabilities = logits
        
        # Stitch them together into a perfect (8400, 13) matrix
        predictions = np.concatenate([boxes, probabilities], axis=1)
        
        max_conf = np.max(predictions[:, 4:])
        print(f"\n--- MATRIX X-RAY ---")
        print(f"Final Matrix Shape: {predictions.shape}")
        print(f"Highest confidence score: {max_conf:.4f}")
        print(f"--------------------\n")

        print("Decoding Bounding Boxes...")
        classes = [
            "bottle-glass", "bottle-plastic", "cup-disposable", 
            "cup-handle", "glass-mug", "glass-normal", 
            "glass-wine", "gym bottle", "tin can"
        ]
        
        final_boxes = []
        final_confs = []
        class_ids = []

        for row in predictions:
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]
            
            # Confidence threshold
            if conf > 0.25:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Scale up if coordinates are somehow normalized [0.0-1.0]
                if w < 2.0 and h < 2.0:
                    cx, cy, w, h = cx * 640, cy * 640, w * 640, h * 640
                    
                x = int(cx - (w / 2))
                y = int(cy - (h / 2))
                
                final_boxes.append([x, y, int(w), int(h)])
                final_confs.append(float(conf))
                class_ids.append(int(class_id))
                
        boxes_drawn = 0
        if len(final_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(final_boxes, final_confs, 0.25, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = final_boxes[i]
                    conf = final_confs[i]
                    c_id = class_ids[i]
                    label = classes[c_id] if c_id < len(classes) else "Unknown"
                    
                    # Draw the bounding box
                    cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_resized, f"{label} {conf:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    boxes_drawn += 1
                                
        cv2.imwrite(OUTPUT_IMAGE, img_resized)
        print(f"Total objects drawn: {boxes_drawn}")
        print(f"Inference Complete! Output saved to {OUTPUT_IMAGE}")
