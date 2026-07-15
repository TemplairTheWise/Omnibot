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

# ── Tunables ──────────────────────────────────────────────────────────────────
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 640
CAMERA_FPS      = 12
DET_EVERY       = 3     # run detection every N depth frames
DET_CONF_THRESH = 0.40

# ── HEF paths ─────────────────────────────────────────────────────────────────
_MODELS = Path(__file__).parent / "resources" / "models"


def _find_hef(filename: str) -> Path:
    for arch in ("hailo8l", "hailo8"):
        p = _MODELS / arch / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found under {_MODELS}")


def _find_det_hef() -> Path:
    # yolov8s for hailo8l, yolov8m for hailo8; yolov6n as fallback
    for name, arch in [
        ("yolov8s.hef", "hailo8l"),
        ("yolov8m.hef", "hailo8"),
        ("yolov6n.hef", "hailo8l"),
    ]:
        p = _MODELS / arch / name
        if p.exists():
            return p
    raise FileNotFoundError("No detection HEF found in resources/models/")


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


def _parse_detections(tensors: dict, labels: list[str]) -> list[Detection]:
    """
    Parse yolov8s Hailo NMS output into Detection namedtuples.

    HailoRT returns the NMS result as a nested Python list (not a rectangular
    ndarray) because each class may have a different number of detections:

        tensors[key]                    # outer list, length = batch size
        tensors[key][0]                 # batch 0, length = num_classes (80)
        tensors[key][0][class_id]       # list of detections for that class
        tensors[key][0][class_id][i]    # one detection: [x1, y1, x2, y2, score]

    Coordinates are normalised to [0, 1] relative to the model input size.
    """
    if not tensors:
        return []

    nms_key = next(
        (k for k in tensors if "nms" in k.lower()),
        next(iter(tensors)),
    )
    raw = tensors[nms_key]

    # Unwrap batch dimension
    try:
        batch_0 = raw[0]
    except (IndexError, TypeError):
        log.warning("Could not unwrap batch from detection output; skipping.")
        return []

    detections: list[Detection] = []
    num_classes = min(len(batch_0), len(labels))

    for class_id in range(num_classes):
        for det in batch_0[class_id]:
            # det = [x1, y1, x2, y2, score]
            x1, y1, x2, y2, score = (float(v) for v in det)
            if score >= DET_CONF_THRESH:
                detections.append(
                    Detection(labels[class_id], score, x1, y1, x2, y2)
                )
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
                    frame_count = 0
                    log.info("Inference worker ready — depth @ %s, det @ %s",
                             f"{depth_h}×{depth_w}", f"{det_h}×{det_w}")

                    while self.state.running:
                        frame = self.state.get_frame()
                        if frame is None:
                            time.sleep(0.01)
                            continue

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # ── Depth (every frame) ───────────────────────────
                        depth_input = cv2.resize(rgb, (depth_w, depth_h))
                        depth_batch = np.expand_dims(depth_input, axis=0)  # (1, H, W, 3)

                        with depth_group.activate(depth_group_params):
                            depth_out = depth_pipe.infer(
                                {depth_in_info.name: depth_batch}
                            )
                        self.state.put_depth(_parse_depth(depth_out))

                        # ── Detection (every DET_EVERY frames) ───────────
                        if frame_count % DET_EVERY == 0:
                            det_input = cv2.resize(rgb, (det_w, det_h))
                            det_batch = np.expand_dims(det_input, axis=0)

                            with det_group.activate(det_group_params):
                                det_out = det_pipe.infer(
                                    {det_in_info.name: det_batch}
                                )
                            self.state.put_detections(
                                _parse_detections(det_out, self.labels)
                            )

                        frame_count += 1

        log.info("InferenceWorker stopped.")


# ── Public interface ──────────────────────────────────────────────────────────
class InferencePipeline:
    """
    Owns the camera reader and inference worker threads.

    get_state() returns a snapshot of the latest results; safe to call from
    any thread at any time (returns None values until the first frames arrive).
    """

    def __init__(self, labels: list[str] = COCO_LABELS):
        self.state   = SharedState()
        self._camera = _CameraReader(self.state)
        self._worker = _InferenceWorker(self.state, labels)

    def start(self) -> None:
        log.info("Starting InferencePipeline …")
        self.state.running = True
        self._camera.start()
        self._worker.start()

    def stop(self) -> None:
        log.info("Stopping InferencePipeline …")
        self.state.running = False
        self._camera.join(timeout=5)
        self._worker.join(timeout=5)

    def get_state(self) -> tuple[np.ndarray | None, np.ndarray | None, list[Detection]]:
        """Return (bgr_frame, depth_map, detections). All None until first frame."""
        return self.state.snapshot()


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
