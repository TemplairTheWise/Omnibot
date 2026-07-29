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
)


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
