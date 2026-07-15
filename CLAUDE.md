# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Before running any Python scripts, source the environment script from the project root:
```bash
source setup_env.sh
```
This activates the `venv_hailo_rpi_examples` virtual environment and sets `PYTHONPATH` to the project root.

## Installation

```bash
./install.sh                  # Full install (reads config.yaml, downloads resources)
./install.sh --no-installation # Skip package install, re-download resources only
./install.sh --all            # Download all available model variants
```

## Running Examples

```bash
# Basic pipelines (GStreamer-based, high-level)
python basic_pipelines/detection_simple.py
python basic_pipelines/detection.py --input rpi   # Raspberry Pi camera
python basic_pipelines/detection.py --input usb   # USB webcam
python basic_pipelines/detection.py --input /dev/video0
python basic_pipelines/pose_estimation.py
python basic_pipelines/instance_segmentation.py
python basic_pipelines/depth.py

# Direct HailoRT API (low-level, no GStreamer)
python custom_yolo26.py       # Single image inference with custom YOLO model
python video_yolo26.py        # Live camera inference with custom YOLO model

# Robotics utilities
python drive.py               # Keyboard-controlled OmniBot (WASD + G for gripper)
python distance.py            # HC-SR04 ultrasonic distance sensor readout
python force_test.py          # Single servo channel test
```

## Running Tests

```bash
bash run_tests.sh             # Sources env, installs test deps, downloads HEFs, runs pytest
pytest -s tests/test_hailo_rpi5_examples.py  # Run test suite directly (after sourcing env)
```

## Architecture

### Two inference approaches

**1. GStreamer pipeline approach** (`basic_pipelines/`) — the standard pattern for this repo:
- Imports a pre-built pipeline class from `hailo_apps` (the [`hailo-apps-infra`](https://github.com/hailo-ai/hailo-apps-infra) pip package)
- Defines `user_app_callback_class(app_callback_class)` to carry per-session state
- Defines `app_callback(pad, info, user_data)` — a GStreamer pad probe that runs on every frame
- GStreamer attaches Hailo metadata to each `GstBuffer`; the callback reads it via `hailo.get_roi_from_buffer(buffer)` then traverses `.get_objects_typed(hailo.HAILO_DETECTION)` etc.
- Frame pixel data is available only when `user_data.use_frame` is True (enabled by `--use-frame` flag), via `get_numpy_from_buffer()`
- The callback must return quickly; offload slow work to a background process

**2. Direct HailoRT API approach** (`custom_yolo26.py`, `video_yolo26.py`):
- Uses `hailo_platform` (`VDevice`, `HEF`, `InferVStreams`, etc.) directly over PCIe
- Loads a `.hef` compiled model file, configures vstreams, and calls `infer_pipeline.infer({input_name: batch})`
- Post-processing (NMS, box decoding) is done manually in NumPy/OpenCV
- Camera frames are captured via `rpicam-vid` subprocess piped as raw YUV420

### Robotics layer

`omnibot.py` — `OmniBot` class: controls a 4-wheel omnidirectional robot via PCA9685 PWM driver over I2C. Motors are addressed as Adafruit servo objects; speed is mapped from `[-100, 100]%` to servo angle. Even-indexed motors are mirrored (sign-flipped).

`drive.py` — keyboard teleop over raw terminal input with a heartbeat: if no key is pressed for 12 s, sends a brief 5% forward pulse to prevent servo sleep.

### Configuration

`config.yaml` — controls which versions of `hailo-apps-infra`, HailoRT, TAPPAS, and the model zoo are used, plus architecture detection (`hailo_arch`, `host_arch` both default to `"auto"`). `install.sh` reads this file.

### Resources

`resources/` — populated by `download_resources.sh` / `hailo-download-resources`. Contains:
- `models/` — `.hef` compiled model files
- `videos/` — test video clips
- `json/` — label maps
- `so/` — compiled post-processing shared libraries (TAPPAS)

Custom `.hef` files (`yolo26_split.hef`, `yolo26_native.hef`, etc.) live in the project root and are referenced directly by the custom scripts.

### Community projects

Each subdirectory under `community_projects/` is self-contained with its own `requirements.txt` and `download_resources.sh`. They follow the same GStreamer callback pattern as `basic_pipelines/` but may add background processing threads for heavy work.
