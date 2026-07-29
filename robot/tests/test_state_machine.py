"""Unit tests for NavStateMachine (state_machine.py), using fake OmniBot/pipeline/SonarGuard
so the FSM logic can be exercised without any real hardware."""

import time

import numpy as np
import pytest

from inference_pipeline import Detection
from state_machine import NavStateMachine, IDLE, SEARCHING, APPROACHING, FOUND


class FakeBot:
    def __init__(self):
        self.calls = []

    def startMove(self, vect, speed):
        self.calls.append(("startMove", vect, speed))

    def rotate(self, direction, speed):
        self.calls.append(("rotate", direction, speed))

    def stop(self):
        self.calls.append(("stop",))


class FakeSonar:
    def __init__(self, distance_cm=200.0):
        self.distance_cm = distance_cm
        self.zone = "clear"


class FakePipeline:
    """Returns a fixed depth map and a controllable list of detections."""

    def __init__(self):
        self.depth = np.zeros((60, 80), dtype=np.float32)
        self.detections: list[Detection] = []

    def get_state(self):
        return None, self.depth, list(self.detections)

    def start_recording(self, *a, **kw):
        pass

    def stop_recording(self):
        pass


def _wait_for_state(nsm, target_states, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = nsm.get_status()["state"]
        if state in target_states:
            return state
        time.sleep(0.02)
    return nsm.get_status()["state"]


def test_reaches_found_when_target_box_covers_frame():
    bot = FakeBot()
    pipeline = FakePipeline()
    sonar = FakeSonar(distance_cm=200.0)  # far - arrival must come via box area, not sonar
    pipeline.detections = [
        Detection(label="bottle-plastic", confidence=0.9,
                  x1=0.2, y1=0.1, x2=0.9, y2=0.9)  # area = 0.7*0.8 = 0.56 >= FOUND_BOX_AREA
    ]

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=True)
    assert nsm.start(target_label="bottle-plastic") is True

    final_state = _wait_for_state(nsm, {FOUND})
    assert final_state == FOUND
    assert ("stop",) in bot.calls

    nsm.stop()
    assert nsm.get_status()["state"] == IDLE


def test_start_returns_false_when_already_running():
    bot = FakeBot()
    pipeline = FakePipeline()
    nsm = NavStateMachine(bot, pipeline, sonar=None, skip_scan=True)

    assert nsm.start() is True
    assert nsm.start() is False  # already running - must be rejected

    nsm.stop()


def test_stop_returns_to_idle_while_searching():
    bot = FakeBot()
    pipeline = FakePipeline()   # no detections -> stays in SEARCHING
    nsm = NavStateMachine(bot, pipeline, sonar=None, skip_scan=True)

    nsm.start(target_label="bottle-plastic")
    time.sleep(0.3)
    assert nsm.get_status()["state"] == SEARCHING

    nsm.stop()
    assert nsm.get_status()["state"] == IDLE
    assert ("stop",) in bot.calls
