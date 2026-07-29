"""Unit tests for PolarScan (polar_scan.py), using FakeBot/FakePipeline with fast timing
parameters so the (normally multi-second) 360 deg scan completes almost instantly."""

import numpy as np
import pytest

from inference_pipeline import Detection
from polar_scan import PolarScan, ROTATE_DIR


class FakeBot:
    def __init__(self):
        self.calls = []

    def rotate(self, direction, speed):
        self.calls.append(("rotate", direction, speed))

    def stop(self):
        self.calls.append(("stop",))


class FakePipeline:
    """Returns an open depth map; detections can be scripted per call via `detections_by_call`."""

    def __init__(self, detections_by_call=None):
        self.depth = np.zeros((60, 80), dtype=np.float32)
        self.detections_by_call = detections_by_call or []
        self.call_count = 0

    def get_state(self):
        dets = (
            self.detections_by_call[self.call_count]
            if self.call_count < len(self.detections_by_call)
            else []
        )
        self.call_count += 1
        return None, self.depth, dets


def _fast_scanner(bot, pipeline, n_steps=4, stop_on_target=True):
    return PolarScan(
        bot, pipeline,
        n_steps=n_steps, rotation_speed=100, deg_per_sec=1e6, settle_s=0.0,
        stop_on_target=stop_on_target,
    )


def test_run_completes_all_steps_when_target_never_seen():
    bot = FakeBot()
    pipeline = FakePipeline()  # no detections, ever
    scanner = _fast_scanner(bot, pipeline, n_steps=4)

    result = scanner.run(target_label="bottle-plastic")

    assert result.completed is True
    assert result.target_heading_deg is None
    assert len(result.clearance_map) == 4
    assert result.best_heading_deg in (0.0, 90.0, 180.0, 270.0)
    assert bot.calls.count(("stop",)) >= 4  # one stop per completed step


def test_run_stops_early_when_target_found():
    bot = FakeBot()
    target = Detection(label="bottle-plastic", confidence=0.9, x1=0.4, y1=0.3, x2=0.6, y2=0.9)
    # No detection on the first step, target appears on the second.
    pipeline = FakePipeline(detections_by_call=[[], [target]])
    scanner = _fast_scanner(bot, pipeline, n_steps=8, stop_on_target=True)

    result = scanner.run(target_label="bottle-plastic")

    assert result.completed is False
    assert result.target_heading_deg is not None


def test_run_keeps_scanning_when_stop_on_target_disabled():
    bot = FakeBot()
    target = Detection(label="bottle-plastic", confidence=0.9, x1=0.4, y1=0.3, x2=0.6, y2=0.9)
    pipeline = FakePipeline(detections_by_call=[[target]])
    scanner = _fast_scanner(bot, pipeline, n_steps=4, stop_on_target=False)

    result = scanner.run(target_label="bottle-plastic")

    assert result.completed is True          # ran all steps despite an early sighting
    assert result.target_heading_deg is not None  # but still remembered the target heading


def test_face_heading_picks_shorter_arc_left():
    bot = FakeBot()
    scanner = _fast_scanner(bot, FakePipeline(), n_steps=4)
    scanner.face_heading(270.0)  # > 180 deg away via ROTATE_DIR -> shorter arc is the other way
    rotate_calls = [c for c in bot.calls if c[0] == "rotate"]
    assert rotate_calls
    assert rotate_calls[0][1] != ROTATE_DIR


def test_face_heading_uses_default_direction_for_short_arc():
    bot = FakeBot()
    scanner = _fast_scanner(bot, FakePipeline(), n_steps=4)
    scanner.face_heading(90.0)  # <= 180 deg -> keep the default scan direction
    rotate_calls = [c for c in bot.calls if c[0] == "rotate"]
    assert rotate_calls
    assert rotate_calls[0][1] == ROTATE_DIR


def test_face_heading_skips_rotation_when_already_close():
    bot = FakeBot()
    scanner = _fast_scanner(bot, FakePipeline(), n_steps=4)
    scanner.face_heading(1.0)  # within the 2 deg "close enough" threshold
    assert not any(c[0] == "rotate" for c in bot.calls)
