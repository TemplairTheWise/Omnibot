import cv2
import numpy as np
import sys
import time
import subprocess
import shlex
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
HEF_PATH = Path(__file__).parent / "models" / "yolo26_split.hef"

def read_exact(pipe, size):
    buf = bytearray(size)
    pos = 0
    while pos < size:
        chunk = pipe.read(size - pos)
        if not chunk: return None
        buf[pos:pos+len(chunk)] = chunk
        pos += len(chunk)
    return buf

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

    in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    print("Preparing Inference Pipeline...")
    with InferVStreams(network_group, in_params, out_params) as infer_pipeline:
        
        print("Waking up hardware ISP via rpicam-vid...")
        
        # FIXED: Added -n (no preview) and lowered framerate to prevent tearing
        cmd = "rpicam-vid -n -t 0 --width 640 --height 640 --framerate 12 --codec yuv420 -o -"
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        frame_size = int(640 * 640 * 1.5)
        WINDOW_NAME = 'Hailo YOLO 26 Live Inference'

        print("Starting live video stream. Click the video window, then press 'q' to quit.")

        with network_group.activate(network_group_params):
            while True:
                start_time = time.time()
                
                raw_data = read_exact(process.stdout, frame_size)
                if raw_data is None:
                    print("Camera pipeline closed.")
                    break
                    
                yuv_matrix = np.frombuffer(raw_data, dtype=np.uint8).reshape((960, 640))
                
                frame_rgb = cv2.cvtColor(yuv_matrix, cv2.COLOR_YUV2RGB_I420)
                display_frame = cv2.cvtColor(yuv_matrix, cv2.COLOR_YUV2BGR_I420)
                
                infer_results = infer_pipeline.infer({input_info.name: np.expand_dims(frame_rgb, axis=0)})
                
                out1 = np.squeeze(infer_results[out_infos[0].name])
                out2 = np.squeeze(infer_results[out_infos[1].name])
                
                if out1.shape[0] < out1.shape[-1]: out1 = out1.T
                if out2.shape[0] < out2.shape[-1]: out2 = out2.T
                    
                if out1.shape[1] == 4:
                    boxes, probabilities = out1, out2
                else:
                    boxes, probabilities = out2, out1
                    
                predictions = np.concatenate([boxes, probabilities], axis=1)
                
                classes = [
                    "bottle-glass", "bottle-plastic", "cup-disposable", 
                    "cup-handle", "glass-mug", "glass-normal", 
                    "glass-wine", "gym bottle", "tin can"
                ]
                
                final_boxes, final_confs, class_ids = [], [], []

                for row in predictions:
                    class_scores = row[4:]
                    class_id = np.argmax(class_scores)
                    conf = class_scores[class_id]
                    
                    if conf > 0.40:
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        
                        if w < 2.0 and h < 2.0:
                            cx, cy, w, h = cx * 640, cy * 640, w * 640, h * 640
                            
                        x = int(cx - (w / 2))
                        y = int(cy - (h / 2))
                        
                        final_boxes.append([x, y, int(w), int(h)])
                        final_confs.append(float(conf))
                        class_ids.append(int(class_id))
                        
                if len(final_boxes) > 0:
                    indices = cv2.dnn.NMSBoxes(final_boxes, final_confs, 0.40, 0.4)
                    
                    if len(indices) > 0:
                        console_output = [] # List to hold text for the terminal
                        
                        for i in indices.flatten():
                            x, y, w, h = final_boxes[i]
                            conf = final_confs[i]
                            c_id = class_ids[i]
                            label = classes[c_id] if c_id < len(classes) else "Unknown"
                            
                            # Draw onto the video frame
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{label} {conf:.2f}", (x, y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        
                            # Format the data for the terminal
                            console_output.append(f"{label} ({conf:.2f}) at [x:{x}, y:{y}, w:{w}, h:{h}]")
                            
                        # Print all detected objects for this frame to the console!
                        print(f"Found {len(console_output)} objects: | " + " | ".join(console_output))
                
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC, pressed with the video window focused
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed via the OS close button

        process.terminate()
        cv2.destroyAllWindows()
