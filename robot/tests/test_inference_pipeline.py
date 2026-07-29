"""Unit tests for the pure/plumbing parts of inference_pipeline.py.

_parse_depth / _parse_detections are pure tensor-parsing functions - tested directly
with synthetic numpy tensors, no Hailo hardware involved. InferencePipeline's
start()/stop()/get_state() plumbing and video recording are tested by monkeypatching
_CameraReader.run/_InferenceWorker.run to lightweight fakes, so the real camera
subprocess and the real Hailo VDevice are never touched.
"""

import time

import numpy as np
import pytest

import inference_pipeline as ip
from inference_pipeline import (
    BEVERAGE_LABELS,
    SharedState,
    _parse_depth,
    _parse_detections,
    _read_exact,
)


# ── _read_exact ───────────────────────────────────────────────────────────────

class _ChunkedPipe:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, n):
        return self._chunks.pop(0) if self._chunks else b""


def test_read_exact_assembles_a_buffer_from_multiple_chunks():
    buf = _read_exact(_ChunkedPipe([b"ab", b"cde"]), 5)
    assert bytes(buf) == b"abcde"


def test_read_exact_returns_none_on_stream_close():
    assert _read_exact(_ChunkedPipe([b"ab", b""]), 5) is None


# ── _CameraReader (subprocess mocked out) ─────────────────────────────────────

class _FakeStdout:
    def __init__(self, payloads):
        self._chunks = list(payloads)

    def read(self, n):
        return self._chunks.pop(0) if self._chunks else b""


class _FakeProc:
    def __init__(self, stdout):
        self.stdout = stdout
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_camera_reader_decodes_one_yuv_frame_then_stops(monkeypatch):
    frame_bytes = int(ip.CAMERA_WIDTH * ip.CAMERA_HEIGHT * 1.5)
    fake_proc = _FakeProc(_FakeStdout([bytes(frame_bytes), b""]))
    monkeypatch.setattr(ip.subprocess, "Popen", lambda *a, **kw: fake_proc)

    state = SharedState()
    state.running = True
    reader = ip._CameraReader(state)
    reader.run()

    frame = state.get_frame()
    assert frame is not None
    assert frame.shape == (ip.CAMERA_HEIGHT, ip.CAMERA_WIDTH, 3)
    assert fake_proc.terminated is True


# ── _parse_depth ─────────────────────────────────────────────────────────────

def test_parse_depth_normalises_to_0_1_with_1_as_nearest():
    raw = np.array([[[0.0, 5.0], [10.0, 2.5]]], dtype=np.float32)  # (1, 2, 2) - batch dim
    out = _parse_depth({"depth": raw})
    assert out.shape == (2, 2)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)
    # the raw maximum (10.0) must map to 1.0 (nearest)
    assert out[1, 0] == pytest.approx(1.0)


def test_parse_depth_handles_constant_input_without_dividing_by_zero():
    raw = np.full((1, 4, 4), 0.5, dtype=np.float32)
    out = _parse_depth({"depth": raw})
    assert out.shape == (4, 4)
    assert np.all(out == pytest.approx(0.5))


# ── _parse_detections ─────────────────────────────────────────────────────────

def _make_tensors(n_anchors=20, n_classes=len(BEVERAGE_LABELS)):
    boxes  = np.full((n_anchors, 4), 0.01, dtype=np.float32)
    logits = np.full((n_anchors, n_classes), 0.05, dtype=np.float32)
    return boxes, logits


def test_parse_detections_returns_empty_with_fewer_than_two_tensors():
    assert _parse_detections({"only_one": np.zeros((5, 4))}, BEVERAGE_LABELS, 100, 100) == []


def test_parse_detections_finds_confident_boxes_above_threshold():
    boxes, logits = _make_tensors()
    bottle_idx = BEVERAGE_LABELS.index("bottle-plastic")
    cup_idx    = BEVERAGE_LABELS.index("cup-disposable")

    boxes[0]  = [0.5, 0.5, 0.3, 0.3]   # normalised cx, cy, w, h
    logits[0, bottle_idx] = 0.9

    boxes[1]  = [0.2, 0.2, 0.1, 0.1]
    logits[1, cup_idx] = 0.8

    dets = _parse_detections(
        {"boxes": boxes, "logits": logits}, BEVERAGE_LABELS, input_w=100, input_h=100
    )

    labels = {d.label for d in dets}
    assert labels == {"bottle-plastic", "cup-disposable"}

    bottle = next(d for d in dets if d.label == "bottle-plastic")
    assert bottle.confidence == pytest.approx(0.9, abs=1e-3)
    assert bottle.x1 == pytest.approx(0.35, abs=0.02)
    assert bottle.x2 == pytest.approx(0.65, abs=0.02)


def test_parse_detections_is_order_independent_between_boxes_and_logits():
    boxes, logits = _make_tensors()
    idx = BEVERAGE_LABELS.index("tin can")
    boxes[0]  = [0.5, 0.5, 0.2, 0.2]
    logits[0, idx] = 0.95

    # logits provided first in the dict, boxes second - must still be identified correctly.
    dets = _parse_detections(
        {"logits": logits, "boxes": boxes}, BEVERAGE_LABELS, input_w=100, input_h=100
    )
    assert len(dets) == 1
    assert dets[0].label == "tin can"


def test_parse_detections_filters_low_confidence_anchors():
    boxes, logits = _make_tensors()  # every anchor stays at 0.05, below DET_CONF_THRESH
    dets = _parse_detections({"boxes": boxes, "logits": logits}, BEVERAGE_LABELS, 100, 100)
    assert dets == []


def test_parse_detections_transposes_channel_first_tensors():
    idx = BEVERAGE_LABELS.index("glass-wine")
    boxes  = np.full((4, 20), 0.01, dtype=np.float32)    # channels-first: (4, N)
    logits = np.full((len(BEVERAGE_LABELS), 20), 0.05, dtype=np.float32)  # (C, N)
    boxes[:, 0]  = [0.5, 0.5, 0.2, 0.2]
    logits[idx, 0] = 0.9

    dets = _parse_detections(
        {"boxes": boxes, "logits": logits}, BEVERAGE_LABELS, input_w=100, input_h=100
    )
    assert len(dets) == 1
    assert dets[0].label == "glass-wine"


# ── SharedState ────────────────────────────────────────────────────────────

def test_shared_state_snapshot_is_atomic_and_returns_latest_values():
    state = SharedState()
    assert state.snapshot() == (None, None, [])

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.float32)
    state.put_frame(frame)
    state.put_depth(depth)
    state.put_detections([1, 2, 3])

    f, d, dets = state.snapshot()
    assert f is frame
    assert d is depth
    assert dets == [1, 2, 3]


# ── InferencePipeline plumbing (hardware threads faked out) ──────────────────

def _fake_camera_run(self):
    self.state.put_frame(np.zeros((10, 10, 3), dtype=np.uint8))
    while self.state.running:
        time.sleep(0.01)


def _fake_worker_run(self):
    self.state.put_depth(np.zeros((10, 10), dtype=np.float32))
    self.state.put_detections([])
    while self.state.running:
        time.sleep(0.01)


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(ip._CameraReader, "run", _fake_camera_run)
    monkeypatch.setattr(ip._InferenceWorker, "run", _fake_worker_run)
    pipeline = ip.InferencePipeline()
    pipeline.start()
    yield pipeline
    pipeline.stop()


def _wait_for_frame(pipeline, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, depth, dets = pipeline.get_state()
        if frame is not None and depth is not None:
            return frame, depth, dets
        time.sleep(0.01)
    return pipeline.get_state()


def test_pipeline_start_populates_shared_state(fake_pipeline):
    frame, depth, dets = _wait_for_frame(fake_pipeline)
    assert frame is not None
    assert depth is not None
    assert dets == []


def test_pipeline_recording_writes_a_video_file(fake_pipeline, tmp_path):
    _wait_for_frame(fake_pipeline)
    out_path = tmp_path / "debug.mp4"
    fake_pipeline.start_recording(str(out_path), flip=True)
    time.sleep(0.3)
    fake_pipeline.stop_recording()
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_pipeline_recording_draws_detection_overlays(fake_pipeline, tmp_path):
    from inference_pipeline import Detection

    _wait_for_frame(fake_pipeline)
    fake_pipeline.state.put_detections([
        Detection(label="bottle-plastic", confidence=0.9, x1=0.2, y1=0.2, x2=0.6, y2=0.8)
    ])
    out_path = tmp_path / "with_dets.mp4"
    fake_pipeline.start_recording(str(out_path), flip=False)
    time.sleep(0.3)
    fake_pipeline.stop_recording()
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_pipeline_start_recording_twice_is_a_noop(fake_pipeline, tmp_path):
    _wait_for_frame(fake_pipeline)
    out_path = tmp_path / "debug.mp4"
    fake_pipeline.start_recording(str(out_path))
    first_thread = fake_pipeline._rec_thread
    fake_pipeline.start_recording(str(out_path))  # already recording - must be ignored
    assert fake_pipeline._rec_thread is first_thread
    fake_pipeline.stop_recording()


def test_pipeline_stop_recording_when_not_recording_is_a_noop(fake_pipeline):
    fake_pipeline.stop_recording()  # never started - must not raise


# ── _InferenceWorker.run() with the Hailo SDK fully mocked out ───────────────
#
# This is the one method that talks directly to real Hailo hardware (VDevice,
# HEF, InferVStreams). Every SDK entry point it uses is mocked so the method's
# actual control flow (model loading, dual-network activation switching, the
# depth-every-frame / detection-every-3rd-frame cadence, and cleanup) is
# exercised without ever touching /dev/hailo0.

def test_inference_worker_runs_one_cycle_against_a_mocked_hailo_sdk(mocker):
    boxes, logits = _make_tensors()
    bottle_idx = BEVERAGE_LABELS.index("bottle-plastic")
    boxes[0]  = [0.5, 0.5, 0.2, 0.2]
    logits[0, bottle_idx] = 0.9

    def _vstream_info():
        info = mocker.MagicMock()
        info.shape = (64, 64, 3)
        info.name = "input"
        return info

    mocker.patch("inference_pipeline.HEF", side_effect=lambda p: mocker.MagicMock(
        get_input_vstream_infos=lambda: [_vstream_info()],
        get_output_vstream_infos=lambda: [_vstream_info()],
    ))

    group = mocker.MagicMock()
    group.activate.return_value = mocker.MagicMock()
    device = mocker.MagicMock()
    device.configure.side_effect = lambda hef, params: [group]
    mocker.patch("inference_pipeline.VDevice", return_value=mocker.MagicMock(
        __enter__=lambda self: device, __exit__=lambda self, *a: False,
    ))

    depth_pipe = mocker.MagicMock()
    depth_pipe.infer.return_value = {"out": np.zeros((1, 4, 4, 1), dtype=np.float32)}
    det_pipe = mocker.MagicMock()
    det_pipe.infer.return_value = {"boxes": boxes, "logits": logits}

    def _ctx(pipe):
        cm = mocker.MagicMock()
        cm.__enter__.return_value = pipe
        cm.__exit__.return_value = False
        return cm

    mocker.patch("inference_pipeline.InferVStreams", side_effect=[_ctx(depth_pipe), _ctx(det_pipe)])
    mocker.patch("inference_pipeline.InputVStreamParams")
    mocker.patch("inference_pipeline.OutputVStreamParams")
    mocker.patch("inference_pipeline.ConfigureParams")

    state = SharedState()
    state.running = True
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def _get_frame_once():
        state.running = False  # stop the loop right after this single iteration
        return frame

    state.get_frame = _get_frame_once

    worker = ip._InferenceWorker(state, BEVERAGE_LABELS)
    worker.run()

    _, depth, dets = state.snapshot()
    assert depth is not None
    assert len(dets) == 1
    assert dets[0].label == "bottle-plastic"
    group.activate.assert_called()
    group.wait_for_activation.assert_called()
