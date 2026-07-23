"""
Dual-inference pipeline for VFH + object-goal navigation (Step 1).

Runs two background threads:
  _CameraReader     — reads YUV420 frames from rpicam-vid into SharedState
  _InferenceWorker  — loads scdepthv3 + yolov8s into one VDevice;
                      runs depth on every frame, detection every DET_EVERY frames

Public API:
    pipeline = InferencePipeline()
    pipeline.start()
    frame, depth, detections = pipeline.get_state()
    pipeline.stop()
"""

import logging
import shlex
import subprocess
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
    OutputVStreamParams,
    VDevice,
)

log = logging.getLogger(__name__)

# ── COCO 80-class labels ──────────────────────────────────────────────────────
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

Detection = namedtuple("Detection", ["label", "confidence", "x1", "y1", "x2", "y2"])
# x1/y1/x2/y2 are normalised to [0, 1] relative to frame width/height.

# ── Beverage labels (matches yolo26_split.hef and beverage_labels.json) ──────
BEVERAGE_LABELS = [
    "bottle-glass", "bottle-plastic", "cup-disposable",
    "cup-handle", "glass-mug", "glass-normal",
    "glass-wine", "gym bottle", "tin can",
]

# ── Tunables ──────────────────────────────────────────────────────────────────
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 640
CAMERA_FPS      = 12
DET_EVERY       = 3     # run detection every N depth frames
DET_CONF_THRESH = 0.25  # pre-NMS confidence threshold
NMS_IOU_THRESH  = 0.40  # IoU threshold for NMS suppression

# ── HEF paths ─────────────────────────────────────────────────────────────────
_MODELS = Path(__file__).parent.parent / "hailo" / "resources" / "models"


def _find_hef(filename: str) -> Path:
    for arch in ("hailo8l", "hailo8"):
        p = _MODELS / arch / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found under {_MODELS}")


def _find_det_hef() -> Path:
    # Prefer the custom beverage model in the project root (same as custom_yolo26.py).
    # Fall back to general-purpose models if it's missing.
    custom = Path(__file__).parent / "models" / "yolo26_split.hef"
    if custom.exists():
        return custom
    for name, arch in [
        ("yolov8s.hef", "hailo8l"),
        ("yolov8m.hef", "hailo8"),
    ]:
        p = _MODELS / arch / name
        if p.exists():
            return p
    raise FileNotFoundError("No detection HEF found.")


DEPTH_HEF_PATH = _find_hef("scdepthv3.hef")
DET_HEF_PATH   = _find_det_hef()


# ── Shared state ──────────────────────────────────────────────────────────────
class SharedState:
    """Lock-protected container for the latest camera + inference results."""

    def __init__(self):
        self._lock             = threading.Lock()
        self.running           = False   # written from main thread, read by workers
        self.latest_frame: np.ndarray | None     = None  # (H, W, 3) BGR uint8
        self.latest_depth: np.ndarray | None     = None  # (H, W) float32, 0=far 1=near
        self.latest_detections: list[Detection]  = []

    def put_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self.latest_frame = frame

    def put_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self.latest_depth = depth

    def put_detections(self, dets: list[Detection]) -> None:
        with self._lock:
            self.latest_detections = dets

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.latest_frame

    def snapshot(self) -> tuple:
        """Atomically return (frame, depth, detections)."""
        with self._lock:
            return (
                self.latest_frame,
                self.latest_depth,
                list(self.latest_detections),
            )


# ── Camera reader ─────────────────────────────────────────────────────────────
def _read_exact(pipe, size: int) -> bytearray | None:
    buf = bytearray(size)
    pos = 0
    while pos < size:
        chunk = pipe.read(size - pos)
        if not chunk:
            return None
        buf[pos: pos + len(chunk)] = chunk
        pos += len(chunk)
    return buf


class _CameraReader(threading.Thread):
    """Reads YUV420 frames from rpicam-vid; converts to BGR and stores in SharedState."""

    def __init__(self, state: SharedState):
        super().__init__(daemon=True, name="CameraReader")
        self.state = state

    def run(self):
        frame_bytes = int(CAMERA_WIDTH * CAMERA_HEIGHT * 1.5)
        cmd = (
            f"rpicam-vid -n -t 0 "
            f"--width {CAMERA_WIDTH} --height {CAMERA_HEIGHT} "
            f"--framerate {CAMERA_FPS} --codec yuv420 -o -"
        )
        log.info("Camera command: %s", cmd)
        proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        try:
            while self.state.running:
                raw = _read_exact(proc.stdout, frame_bytes)
                if raw is None:
                    log.warning("Camera stream closed.")
                    break
                yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (int(CAMERA_HEIGHT * 1.5), CAMERA_WIDTH)
                )
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                self.state.put_frame(bgr)
        finally:
            proc.terminate()
            log.info("CameraReader stopped.")


# ── Output parsers ────────────────────────────────────────────────────────────
def _parse_depth(tensors: dict) -> np.ndarray:
    """
    scdepthv3 outputs inverse depth (disparity): higher = closer to camera.
    We squeeze batch/channel dims and normalise to [0, 1] so that
    1 = nearest (most dangerous) and 0 = furthest.
    """
    tensor = next(iter(tensors.values()))
    d = np.squeeze(tensor).astype(np.float32)  # → (H, W)
    lo, hi = d.min(), d.max()
    if hi > lo:
        d = (d - lo) / (hi - lo)
    return d


def _parse_detections(
    tensors: dict, labels: list[str], input_w: int, input_h: int
) -> list[Detection]:
    """
    Parse the custom yolo26_split model output into Detection namedtuples.

    The model outputs two raw tensors — no built-in NMS:
      • boxes tensor:  (8400, 4)  — cx, cy, w, h in absolute pixels (or normalised)
      • logits tensor: (8400, N)  — raw class scores (one column per label)

    We apply argmax + confidence filtering + cv2 NMS, then normalise
    the surviving boxes to [0, 1] so downstream code is coordinate-agnostic.
    """
    if len(tensors) < 2:
        log.warning("Expected 2 output tensors from custom model, got %d.", len(tensors))
        return []

    values = list(tensors.values())
    out1 = np.squeeze(values[0]).astype(np.float32)
    out2 = np.squeeze(values[1]).astype(np.float32)

    # Ensure (8400, C) orientation — transpose if channels are on axis 0
    if out1.ndim == 2 and out1.shape[0] < out1.shape[1]:
        out1 = out1.T
    if out2.ndim == 2 and out2.shape[0] < out2.shape[1]:
        out2 = out2.T

    # Assign: whichever has 4 columns is boxes, the other is class logits
    if out1.shape[1] == 4:
        boxes, logits = out1, out2
    else:
        boxes, logits = out2, out1

    # Vectorised: best class and its score for every anchor
    class_ids = np.argmax(logits, axis=1)
    confs      = logits[np.arange(len(logits)), class_ids]

    # Pre-NMS filter
    keep = confs >= DET_CONF_THRESH
    if not np.any(keep):
        return []

    boxes_f    = boxes[keep]
    confs_f    = confs[keep]
    class_ids_f = class_ids[keep]

    # cx,cy,w,h → x,y,w,h  (scale up if coordinates are normalised [0,1])
    cx, cy, bw, bh = boxes_f[:, 0], boxes_f[:, 1], boxes_f[:, 2], boxes_f[:, 3]
    if np.median(bw) < 2.0:          # normalised — scale to pixels
        cx, cy, bw, bh = cx * input_w, cy * input_h, bw * input_w, bh * input_h
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1).astype(int).tolist()

    # NMS
    indices = cv2.dnn.NMSBoxes(xywh, confs_f.tolist(), DET_CONF_THRESH, NMS_IOU_THRESH)
    if not len(indices):
        return []

    detections: list[Detection] = []
    for i in indices.flatten():
        x, y, w_px, h_px = xywh[i]
        x1 = max(0.0, x / input_w)
        y1 = max(0.0, y / input_h)
        x2 = min(1.0, (x + w_px) / input_w)
        y2 = min(1.0, (y + h_px) / input_h)
        cid = int(class_ids_f[i])
        label = labels[cid] if cid < len(labels) else "unknown"
        detections.append(Detection(label, float(confs_f[i]), x1, y1, x2, y2))
    return detections


# ── Inference worker ──────────────────────────────────────────────────────────
class _InferenceWorker(threading.Thread):
    """
    Loads both HEFs into one VDevice.
    Alternates activation: depth every frame, detection every DET_EVERY frames.
    InferVStreams are created once and held open for the session lifetime.
    """

    def __init__(self, state: SharedState, labels: list[str]):
        super().__init__(daemon=True, name="InferenceWorker")
        self.state  = state
        self.labels = labels

    def run(self):  # noqa: C901 – intentionally linear setup flow
        log.info("Loading depth HEF:     %s", DEPTH_HEF_PATH)
        log.info("Loading detection HEF: %s", DET_HEF_PATH)
        depth_hef = HEF(str(DEPTH_HEF_PATH))
        det_hef   = HEF(str(DET_HEF_PATH))

        with VDevice() as device:
            # Configure both networks into the same VDevice
            depth_group = device.configure(
                depth_hef,
                ConfigureParams.create_from_hef(
                    hef=depth_hef, interface=HailoStreamInterface.PCIe
                ),
            )[0]
            det_group = device.configure(
                det_hef,
                ConfigureParams.create_from_hef(
                    hef=det_hef, interface=HailoStreamInterface.PCIe
                ),
            )[0]

            # Inspect and log stream shapes (useful for verifying output format)
            depth_in_info  = depth_hef.get_input_vstream_infos()[0]
            det_in_info    = det_hef.get_input_vstream_infos()[0]

            log.info("Depth  input  shape: %s  name: %s",
                     depth_in_info.shape, depth_in_info.name)
            for o in depth_hef.get_output_vstream_infos():
                log.info("Depth  output shape: %s  name: %s", o.shape, o.name)
            log.info("Det    input  shape: %s  name: %s",
                     det_in_info.shape, det_in_info.name)
            for o in det_hef.get_output_vstream_infos():
                log.info("Det    output shape: %s  name: %s", o.shape, o.name)

            # shape is (H, W, C) for UINT8 NHWC inputs
            depth_h, depth_w = depth_in_info.shape[0], depth_in_info.shape[1]
            det_h,   det_w   = det_in_info.shape[0],   det_in_info.shape[1]

            depth_in_params  = InputVStreamParams.make(depth_group, format_type=FormatType.UINT8)
            depth_out_params = OutputVStreamParams.make(depth_group, format_type=FormatType.FLOAT32)
            det_in_params    = InputVStreamParams.make(det_group,   format_type=FormatType.UINT8)
            det_out_params   = OutputVStreamParams.make(det_group,  format_type=FormatType.FLOAT32)

            depth_group_params = depth_group.create_params()
            det_group_params   = det_group.create_params()

            with InferVStreams(depth_group, depth_in_params, depth_out_params) as depth_pipe:
                with InferVStreams(det_group, det_in_params, det_out_params) as det_pipe:
                    frame_count  = 0
                    active_name  = None  # "depth" | "det" | None
                    active_ctx   = None  # entered ActivatedNetworkContextManager
                    log.info("Inference worker ready — depth @ %s, det @ %s",
                             f"{depth_h}×{depth_w}", f"{det_h}×{det_w}")

                    def _switch_to(group, group_params, name):
                        # Both HEFs share one VDevice, so only one group may be
                        # active at a time. Re-entering activate() on every
                        # frame — even when the group hasn't changed — thrashes
                        # the device and races HailoRT's async activation,
                        # producing HAILO_STREAM_NOT_ACTIVATED read failures.
                        # Only switch (and wait for activation to land) when
                        # the required group actually changes.
                        nonlocal active_name, active_ctx
                        if active_name == name:
                            return
                        if active_ctx is not None:
                            active_ctx.__exit__(None, None, None)
                        active_ctx = group.activate(group_params)
                        active_ctx.__enter__()
                        group.wait_for_activation()
                        active_name = name

                    try:
                        while self.state.running:
                            frame = self.state.get_frame()
                            if frame is None:
                                time.sleep(0.01)
                                continue

                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # ── Depth (every frame) ───────────────────────
                            depth_input = cv2.resize(rgb, (depth_w, depth_h))
                            depth_batch = np.expand_dims(depth_input, axis=0)  # (1, H, W, 3)

                            _switch_to(depth_group, depth_group_params, "depth")
                            depth_out = depth_pipe.infer(
                                {depth_in_info.name: depth_batch}
                            )
                            self.state.put_depth(_parse_depth(depth_out))

                            # ── Detection (every DET_EVERY frames) ────────
                            if frame_count % DET_EVERY == 0:
                                det_input = cv2.resize(rgb, (det_w, det_h))
                                det_batch = np.expand_dims(det_input, axis=0)

                                _switch_to(det_group, det_group_params, "det")
                                det_out = det_pipe.infer(
                                    {det_in_info.name: det_batch}
                                )
                                self.state.put_detections(
                                    _parse_detections(det_out, self.labels, det_w, det_h)
                                )

                            frame_count += 1
                    finally:
                        if active_ctx is not None:
                            active_ctx.__exit__(None, None, None)

        log.info("InferenceWorker stopped.")


# ── Public interface ──────────────────────────────────────────────────────────
class InferencePipeline:
    """
    Owns the camera reader and inference worker threads.

    get_state() returns a snapshot of the latest results; safe to call from
    any thread at any time (returns None values until the first frames arrive).
    """

    def __init__(self, labels: list[str] = BEVERAGE_LABELS):
        self.state   = SharedState()
        self._camera = _CameraReader(self.state)
        self._worker = _InferenceWorker(self.state, labels)
        self._rec_thread: threading.Thread | None = None
        self._rec_running = False
        self._rec_flip    = False

    def start(self) -> None:
        log.info("Starting InferencePipeline …")
        self.state.running = True
        self._camera.start()
        self._worker.start()

    def stop(self) -> None:
        log.info("Stopping InferencePipeline …")
        self.stop_recording()
        self.state.running = False
        self._camera.join(timeout=5)
        self._worker.join(timeout=5)

    def get_state(self) -> tuple[np.ndarray | None, np.ndarray | None, list[Detection]]:
        """Return (bgr_frame, depth_map, detections). All None until first frame."""
        return self.state.snapshot()

    # ── Video recording ───────────────────────────────────────────────────────

    def start_recording(self, path: str = "debug_approach.mp4",
                        flip: bool = False) -> None:
        """
        Begin saving annotated frames (detection boxes + labels) to a video file.
        Call stop_recording() or stop() to flush and close.

        Parameters
        ----------
        path : output .mp4 path
        flip : rotate the saved frames 180° for viewing when the camera is
               mounted upside-down.  Does not affect inference — bearing
               calculation always uses the original orientation.
        """
        if self._rec_running:
            log.warning("Already recording — stop first.")
            return
        self._rec_flip = flip
        self._rec_running = True
        self._rec_thread = threading.Thread(
            target=self._record_loop, args=(path,),
            name="VideoRecorder", daemon=True,
        )
        self._rec_thread.start()
        log.info("Recording started → %s  flip=%s", path, flip)

    def stop_recording(self) -> None:
        """Flush and close the video file."""
        if not self._rec_running:
            return
        self._rec_running = False
        if self._rec_thread:
            self._rec_thread.join(timeout=5.0)
        log.info("Recording stopped.")

    def _record_loop(self, path: str) -> None:
        writer = None
        prev_frame_id = None
        try:
            while self._rec_running:
                frame, _, dets = self.get_state()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Skip if frame hasn't changed since last write
                frame_id = id(frame)
                if frame_id == prev_frame_id:
                    time.sleep(1.0 / CAMERA_FPS)
                    continue
                prev_frame_id = frame_id

                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(path, fourcc, CAMERA_FPS, (w, h))
                    log.info("VideoWriter opened: %dx%d @ %d fps", w, h, CAMERA_FPS)

                vis = frame.copy()
                h, w = vis.shape[:2]

                for d in dets:
                    x1 = int(d.x1 * w);  y1 = int(d.y1 * h)
                    x2 = int(d.x2 * w);  y2 = int(d.y2 * h)
                    cx = (x1 + x2) // 2
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{d.label} {d.confidence:.0%}"
                    cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    # Bearing line from bottom-centre to detection centroid
                    cv2.line(vis, (w // 2, h), (cx, (y1 + y2) // 2),
                             (0, 200, 255), 1)

                # Frame centre crosshair
                cv2.line(vis, (w//2 - 15, h//2), (w//2 + 15, h//2), (255,255,255), 1)
                cv2.line(vis, (w//2, h//2 - 15), (w//2, h//2 + 15), (255,255,255), 1)

                if self._rec_flip:
                    vis = cv2.rotate(vis, cv2.ROTATE_180)

                writer.write(vis)
                time.sleep(1.0 / CAMERA_FPS)

        except Exception:
            log.exception("VideoRecorder error")
        finally:
            if writer:
                writer.release()


# ── Goal-bearing helper ───────────────────────────────────────────────────────

def goal_bearing_from_detections(
    detections: list[Detection],
    target_label: str | None = None,
    camera_fov_deg: float = 62.0,
) -> float | None:
    """
    Compute the horizontal bearing to the best matching detection.

    Parameters
    ----------
    detections     : list from InferencePipeline.get_state()
    target_label   : if given, only detections with this label are considered;
                     if None, any detection is eligible
    camera_fov_deg : horizontal field of view of the camera

    Returns
    -------
    float  — bearing in degrees (+ = right, − = left, 0 = straight ahead)
    None   — no matching detection in the current frame

    Selection strategy: the candidate with the largest bounding-box area is
    chosen (largest box = closest to camera = highest priority target).
    """
    candidates = [d for d in detections
                  if target_label is None or d.label == target_label]
    if not candidates:
        return None

    best = max(candidates, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
    cx = (best.x1 + best.x2) / 2.0          # normalised [0, 1]
    return (cx - 0.5) * camera_fov_deg       # degrees; + = right of centre


# ── Smoke-test entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    )

    pipeline = InferencePipeline()
    pipeline.start()

    log.info("Running for 10 seconds — watch for depth/detection log output …")
    try:
        for _ in range(10):
            time.sleep(1)
            frame, depth, dets = pipeline.get_state()
            depth_summary = (
                f"min={depth.min():.2f} max={depth.max():.2f} mean={depth.mean():.2f}"
                if depth is not None else "not ready"
            )
            log.info(
                "frame=%s  depth=%s  detections=%d",
                "ok" if frame is not None else "none",
                depth_summary,
                len(dets),
            )
            for d in dets:
                log.info("  › %s  conf=%.2f  box=(%.2f,%.2f,%.2f,%.2f)",
                         d.label, d.confidence, d.x1, d.y1, d.x2, d.y2)
    finally:
        pipeline.stop()
