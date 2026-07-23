---
name: project-navigation-implementation
description: Status of the VFH + object-goal navigation implementation for the OmniBot
metadata:
  type: project
---

Building VFH + object-goal navigation on top of the Hailo RPi5 robot setup. Implementation plan divided into 11 steps.

**Why:** Robot needs to autonomously search for and approach beverage containers using a single forward-facing camera (no LIDAR, no odometry).

**How to apply:** When continuing implementation, start from the next uncompleted step. Each step builds on the previous.

## Completed steps

### Step 1 — Dual inference pipeline (`inference_pipeline.py`) ✅
- Camera reader thread: `rpicam-vid` YUV420 at 640×640 12fps → BGR frames
- Depth model: `scdepthv3.hef` (hailo8l), input (256,320,3), output (256,320,1) inverse depth → normalised [0,1]
- Detection model: `yolo26_split.hef` (project root, custom beverage model), input (640,640,3), outputs (1,8400,4) boxes + (1,8400,9) logits → manual NMS
- Labels: 9 beverage classes from `beverage_labels.json` / `BEVERAGE_LABELS` constant
- Public API: `InferencePipeline().start()` / `.stop()` / `.get_state()` → (frame, depth, detections)
- All thread-safe via `SharedState` with a single lock

### Step 2 — VFH module (`vfh.py`) ✅
- Divides depth image into N sectors, takes max depth per sector, inverts to clearance
- Ground crop (default 30% bottom) removes floor from normalization
- Gaussian smoothing, valley detection, goal-biased valley scoring
- Returns `VFHResult(angle_deg, vx, vy, clearance, valley_mask)`

### Step 3 — Navigator (`navigator.py`) ✅
- Translates VFH output to OmniBot commands
- Angles > 20° → rotate in place; smaller angles → `startMove`
- Speed scaled by fraction of passable sectors (BASE_SPEED=80%, MIN_SPEED=35%)

### Step 4 — Sonar safety guard (`sonar_guard.py`) ✅
- Daemon thread polls HC-SR04 at 10 Hz (GPIO 5/6 via gpiozero)
- Publishes `zone`: CLEAR (≥60 cm) / SLOW (20-60 cm) / STOP (<20 cm)
- Navigator checks zone at start of every `step()`: STOP → bot.stop()+return None; SLOW → cap speed to SLOW_ZONE_CAP (45%)
- Optional: pass `sonar=SonarGuard()` to Navigator; standalone `python navigator.py --sonar`

### Step 5 — Goal bearing from detections ✅
- `goal_bearing_from_detections(detections, target_label, camera_fov_deg)` in `inference_pipeline.py`
- Picks the largest bounding box (closest target), computes bearing: `(cx - 0.5) * fov`
- `Navigator.run()` calls it each frame; falls back to fixed `goal_bearing_deg` (default 0°) when nothing visible
- Standalone: `python navigator.py --target bottle-plastic --sonar`
- Console shows `[locked]` / `[no target]` state each frame

### Step 6 — 360° polar scan (`polar_scan.py`) ✅
- 18 steps × 20° = 360°; each step: rotate (timed) → stop → settle 0.4s → sample depth + detections
- Produces `ScanResult`: clearance_map (n_steps,), best_heading_deg, target_heading_deg (or None)
- `stop_on_target=True`: aborts scan early when target detected, robot left facing it
- `face_heading(deg)`: rotates robot to a global heading using shorter arc
- DEG_PER_SEC must be calibrated on hardware; dry run with `--no-move` flag
- ASCII clearance bar chart printed in standalone mode

### Step 7 — Navigation state machine (`state_machine.py`) ✅
- States: IDLE → SCANNING → SEARCHING → APPROACHING → FOUND
- Runs in daemon thread; thread-safe start()/stop()/get_status() for Flask
- SCANNING: runs PolarScan; target found → APPROACHING, else face best_heading → SEARCHING
- SEARCHING: VFH avoidance at goal=0°; detection → APPROACHING
- APPROACHING: VFH with target bearing; sonar STOP + visible OR box area ≥ 12% → FOUND; N lost frames → SEARCHING
- skip_scan=True skips polar scan; standalone: python state_machine.py --target bottle-plastic

### Step 8 — Flask `/search` endpoints ✅
- `POST /search/start` → `{"target": ..., "scan": false, "record": "out.mp4", "flip": true}`; calls `nsm.start(skip_scan=not scan)`. Returns 409 if already running.
- `POST /search/stop` → calls `nsm.stop()`
- `GET /search/status` → returns `nsm.get_status()` (state, target, detected, confidence, bearing_deg, sonar_cm)
- `GET /search/labels` → returns BEVERAGE_LABELS list
- Navigation pipeline (InferencePipeline + SonarGuard + NavStateMachine) initialized at server startup in try/except; server still works for manual control if Hailo unavailable
- Manual move/rotate endpoints auto-stop the NSM before issuing drive commands; `/stop` always stops both NSM and bot
- Watchdog suppressed during autonomous navigation: calls `_touch()` each 0.25s cycle when NSM is active

### Step 9 — Browser UI search panel ✅
- Search panel added below keyboard legend in `robot_server.py` HTML
- Target label dropdown auto-populated from `/search/labels`
- Polar scan checkbox, record path input, flip checkbox
- Start / Stop buttons
- Status box polls `/search/status` every 1 second: state badge (color-coded per state), target, detected+confidence, bearing, sonar distance

### Step 11 — Session logging (`state_machine.py`) ✅
- `LOG_DIR = Path("search_logs")` — JSON file per session
- `_log_event(kind, **kw)` — appends to `_session["events"]` list with monotonic timestamp
- `_finalize_session(outcome)` — writes JSON on run-thread exit; called in finally block
- `get_history(limit=20)` — reads session JSONs from disk, returns list of dicts
- `_set_state()` now also calls `_log_event("state", state=new_state)`
- Detection events logged in `_searching_phase()` with label, confidence, bearing, state

### Step 12 — Periodic re-scan (`state_machine.py`) ✅
- `RESCAN_AFTER_S = 20.0` — re-scan if no detection for this long in SEARCHING
- `_inline_rescan()` — stops bot, runs PolarScan inline, updates state (SCANNING → APPROACHING or SEARCHING)
- Triggers in `_searching_phase()` when `time.monotonic() - last_search_det_t > RESCAN_AFTER_S`
- Re-scan timer reset after `_inline_rescan()` and when target lost (APPROACHING → SEARCHING)

### Step 13 — Detection evaluation (`eval_detection.py`) ✅
- `python eval_detection.py --images /path/to/dataset/`
- Loads `yolo26_split.hef` directly via VDevice; mirrors `inference_pipeline.py` API (UINT8 input, `group.activate(params)` context)
- PASCAL VOC all-point interpolated AP per class; mAP@0.5 overall
- Outputs per-class: AP, precision, recall, F1, TP count, GT count
- Latency: mean and p95 in ms; results written to `eval_detection_results.json`
- Skips images with no matching `.txt` annotation file

### Step 14 — Navigation evaluation (`eval_navigation.py`) ✅
- `python eval_navigation.py [--trials N] [--scan] [--timeout 120]`
- Interactive: per-trial prompts for scenario, target, record path, success confirmation, notes
- Polls NSM status during trial; Ctrl-C aborts trial cleanly
- Reads detections_count from `nsm.get_history()` after each trial
- Appends CSV rows to `eval_navigation_results.csv` (accumulates across sessions)
- Session summary table printed at end

## Remaining steps
10. Tune constants on hardware (APPROACH_ROTATE_BEARING, TARGET_LOST_FRAMES, FOUND_BOX_AREA, DEG_PER_SEC)
