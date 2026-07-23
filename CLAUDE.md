# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Before running any Python scripts, source the environment script from the project root:
```bash
source setup_env.sh
```
This activates the `venv_hailo_rpi_examples` virtual environment and sets `PYTHONPATH` to both the project root and `robot/`.

## Installation (Hailo SDK + models)

```bash
bash hailo/install.sh                   # Full install (reads hailo/config.yaml)
bash hailo/install.sh --no-installation # Re-download resources only
bash hailo/install.sh --all             # Download all model variants
```

## Repository layout

```
omnibot/
├── robot/                        ← all custom thesis work
│   ├── omnibot.py                  ← OmniBot motor control (PCA9685 / I2C)
│   ├── drive.py                    ← keyboard teleop (WASD + G)
│   ├── distance.py                 ← HC-SR04 sonar diagnostic
│   ├── force_test.py               ← servo channel diagnostic
│   ├── robot_server.py             ← Flask API + web UI (port 5000)
│   ├── inference_pipeline.py       ← camera reader + dual Hailo inference
│   ├── vfh.py                      ← Vector Field Histogram algorithm
│   ├── navigator.py                ← VFH → OmniBot drive commands
│   ├── polar_scan.py               ← 360° clearance scan
│   ├── sonar_guard.py              ← HC-SR04 safety zone daemon
│   ├── state_machine.py            ← autonomous nav FSM
│   ├── beverage_labels.json        ← label map for the custom YOLO model
│   ├── custom_yolo26.py            ← single-image inference prototype
│   ├── video_yolo26.py             ← live-camera inference prototype
│   ├── models/
│   │   ├── yolo26_split.hef        ← custom beverage detector (production)
│   │   ├── yolo26_perfected.hef
│   │   ├── yolo26_native.hef
│   │   └── yolo26_custom_compiled.hef
│   ├── eval/
│   │   ├── capture_dataset.py      ← interactive YOLO dataset capture (SPACE to save)
│   │   ├── eval_detection.py       ← mAP@0.5 evaluation
│   │   └── eval_navigation.py      ← navigation trial runner → CSV
│   └── assets/
│       ├── bottles.jpg
│       ├── output_detection.jpg
│       └── debug_approach.mp4
│
├── hailo/                        ← upstream hailo-rpi5-examples (reference only)
│   ├── basic_pipelines/            ← GStreamer detection/segmentation/depth demos
│   ├── community_projects/         ← community add-ons
│   ├── tests/                      ← upstream test suite
│   ├── doc/                        ← upstream documentation
│   ├── resources → /usr/local/hailo/resources   ← symlink to Hailo system models
│   ├── install.sh                  ← downloads SDK + models
│   ├── config.yaml                 ← hailo-apps-infra / HailoRT versions
│   └── requirements.txt
│
├── setup_env.sh                  ← activate venv + set PYTHONPATH (source this first)
├── CLAUDE.md                     ← this file
└── .gitignore
```

## Running the robot

```bash
# Web control interface (autonomous + manual tabs)
python robot/robot_server.py
# → open http://<pi-ip>:5000

# Keyboard teleop only
python robot/drive.py

# Hardware diagnostics
python robot/distance.py      # sonar readout
python robot/force_test.py    # servo test

# Inference prototypes
python robot/custom_yolo26.py  # single image → robot/assets/output_detection.jpg
python robot/video_yolo26.py   # live camera
```

## Evaluation workflow

```bash
# 1. Collect a labelled test set (SPACE to save, Q to quit)
python robot/eval/capture_dataset.py --out dataset/ --flip

# 2. Evaluate detection accuracy
python robot/eval/eval_detection.py --images dataset/
#    → mAP@0.5 table + eval_detection_results.json

# 3. Record navigation trials
python robot/eval/eval_navigation.py --trials 10 --scan
#    → eval_navigation_results.csv
```

## Running the upstream Hailo tests

```bash
bash hailo/run_tests.sh
# or directly:
pytest -s hailo/tests/test_hailo_rpi5_examples.py
```

## Architecture

### Inference pipeline (`robot/inference_pipeline.py`)

Two background threads share a `SharedState`:
- `_CameraReader` — `rpicam-vid` YUV420 → BGR at 640×640 12 fps
- `_InferenceWorker` — both HEFs in one VDevice:
  - depth every frame (`scdepthv3.hef` from `hailo/resources/models/hailo8l/`)
  - detection every 3 frames (`robot/models/yolo26_split.hef`)

Public: `pipeline.get_state()` → `(bgr_frame, depth_map, detections)`

### Navigation stack

```
SonarGuard       — HC-SR04 daemon: CLEAR / SLOW / STOP zones
VFH (vfh.py)     — depth map → clearance histogram → steering angle
Navigator        — VFH + sonar → OmniBot drive commands
PolarScan        — 18 × 20° rotation; maps free headings + target bearing
NavStateMachine  — orchestrates all above; logs sessions to search_logs/
```

States: `IDLE → SCANNING → SEARCHING → APPROACHING → FOUND`

Key tuning constants in `robot/state_machine.py`:
- `APPROACH_ROTATE_BEARING` (8°) — bearing threshold before rotating vs. driving straight
- `FOUND_BOX_AREA` (0.25) — fraction of frame the target must fill to declare "arrived"
- `TARGET_LOST_FRAMES` (50) — frames without detection before APPROACHING → SEARCHING
- `RESCAN_AFTER_S` (20.0) — seconds without detection before a fresh polar scan

### OmniBot (`robot/omnibot.py`)

4-wheel omnidirectional via PCA9685 / I2C. Camera is 180° rotated — bearing from detection must be negated before issuing drive commands.

### Flask server (`robot/robot_server.py`) — port 5000

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/move` | `{x, y, speed}` |
| POST | `/rotate` | `{direction, speed}` |
| POST | `/stop` | Emergency stop |
| POST | `/gripper` | `{open}` or `{toggle}` |
| POST | `/search/start` | `{target, scan, record, flip}` |
| POST | `/search/stop` | |
| GET | `/search/status` | State + telemetry |
| GET | `/search/labels` | Detectable classes |
| GET | `/search/history` | Past session summaries |
