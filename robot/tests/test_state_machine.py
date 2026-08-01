"""Unit tests for NavStateMachine (state_machine.py), using fake OmniBot/pipeline/SonarGuard
so the FSM logic can be exercised without any real hardware."""

import json
import time

import numpy as np
import pytest

import state_machine as sm
from inference_pipeline import Detection
from state_machine import NavStateMachine, IDLE, SCANNING, SEARCHING, APPROACHING, FOUND


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
    """Returns a fixed depth map by default; both depth and detections can be scripted
    per-call via `depth_by_call` / `detections_by_call` (falls back to defaults once
    the scripted list is exhausted)."""

    def __init__(self, depth_by_call=None, detections_by_call=None):
        self.depth = np.zeros((60, 80), dtype=np.float32)
        self.detections: list[Detection] = []
        self.depth_by_call = depth_by_call or []
        self.detections_by_call = detections_by_call or []
        self.call_count = 0

    def get_state(self):
        i = self.call_count
        self.call_count += 1
        depth = self.depth_by_call[i] if i < len(self.depth_by_call) else self.depth
        dets  = self.detections_by_call[i] if i < len(self.detections_by_call) else self.detections
        return None, depth, dets

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


def _wait_until(predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


TARGET = Detection(label="bottle-plastic", confidence=0.9, x1=0.2, y1=0.1, x2=0.9, y2=0.9)


def test_reaches_found_when_target_box_covers_frame():
    bot = FakeBot()
    pipeline = FakePipeline()
    sonar = FakeSonar(distance_cm=200.0)  # far - arrival must come via box area, not sonar
    pipeline.detections = [TARGET]

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


def test_start_with_record_path_starts_pipeline_recording():
    bot = FakeBot()
    pipeline = FakePipeline()
    calls = []
    pipeline.start_recording = lambda path, flip=False: calls.append((path, flip))

    nsm = NavStateMachine(bot, pipeline, sonar=None, skip_scan=True)
    nsm.start(target_label="bottle-plastic", record_path="out.mp4", record_flip=True)
    assert calls == [("out.mp4", True)]
    nsm.stop()


def test_arrival_via_sonar_distance_alone():
    bot = FakeBot()
    pipeline = FakePipeline()
    # Box area is small (below FOUND_BOX_AREA) - arrival must come purely from the sonar.
    pipeline.detections = [Detection(label="bottle-plastic", confidence=0.9,
                                     x1=0.48, y1=0.48, x2=0.52, y2=0.52)]
    sonar = FakeSonar(distance_cm=5.0)  # inside APPROACH_STOP_CM

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {FOUND}) == FOUND
    nsm.stop()


def test_target_lost_for_too_long_returns_to_searching(monkeypatch):
    monkeypatch.setattr(sm, "TARGET_LOST_FRAMES", 2)

    bot = FakeBot()
    # Target visible on the first call (-> APPROACHING), then gone for good.
    pipeline = FakePipeline(detections_by_call=[[TARGET]])
    sonar = FakeSonar(distance_cm=200.0)  # never close enough to "arrive"

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {APPROACHING}) == APPROACHING
    assert _wait_for_state(nsm, {SEARCHING}) == SEARCHING
    nsm.stop()


def test_slow_sonar_zone_caps_approach_speed():
    bot = FakeBot()
    pipeline = FakePipeline()
    pipeline.detections = [Detection(label="bottle-plastic", confidence=0.9,
                                     x1=0.45, y1=0.1, x2=0.55, y2=0.4)]  # small, centred box
    sonar = FakeSonar(distance_cm=20.0)  # inside APPROACH_SLOW_CM, outside APPROACH_STOP_CM

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    def _drove_slow():
        return any(c[0] == "startMove" and c[2] == sm.APPROACH_SLOW_SPEED for c in bot.calls)

    assert _wait_until(_drove_slow)
    nsm.stop()


def test_large_approach_bearing_rotates_instead_of_driving():
    bot = FakeBot()
    pipeline = FakePipeline()
    # Detection far to one side -> large bearing, well above APPROACH_ROTATE_BEARING.
    pipeline.detections = [Detection(label="bottle-plastic", confidence=0.9,
                                     x1=0.85, y1=0.1, x2=0.98, y2=0.3)]
    sonar = FakeSonar(distance_cm=200.0)

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    assert _wait_until(lambda: any(c[0] == "rotate" for c in bot.calls))
    nsm.stop()


def test_depth_none_is_skipped_until_a_real_frame_arrives():
    bot = FakeBot()
    pipeline = FakePipeline(depth_by_call=[None, None])  # falls back to real depth afterwards
    pipeline.detections = [TARGET]

    nsm = NavStateMachine(bot, pipeline, sonar=FakeSonar(distance_cm=200.0), skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {FOUND}, timeout_s=3.0) == FOUND
    nsm.stop()


def test_pipeline_exception_is_caught_and_session_still_finalised(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "LOG_DIR", tmp_path)

    bot = FakeBot()
    pipeline = FakePipeline()

    def _raise():
        raise RuntimeError("simulated pipeline failure")
    pipeline.get_state = _raise

    nsm = NavStateMachine(bot, pipeline, sonar=None, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {IDLE}) == IDLE
    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text())["outcome"] == "TIMEOUT"


def test_scanning_phase_finds_target_transitions_to_approaching():
    bot = FakeBot()
    pipeline = FakePipeline()
    pipeline.detections = [TARGET]

    nsm = NavStateMachine(
        bot, pipeline, sonar=FakeSonar(distance_cm=200.0), skip_scan=False,
        n_steps=2, rotation_speed=100, deg_per_sec=1e6, settle_s=0.0,
    )
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {APPROACHING, FOUND}) in (APPROACHING, FOUND)
    nsm.stop()


def test_scanning_phase_no_target_transitions_to_searching():
    bot = FakeBot()
    pipeline = FakePipeline()  # never any detections

    nsm = NavStateMachine(
        bot, pipeline, sonar=FakeSonar(distance_cm=200.0), skip_scan=False,
        n_steps=2, rotation_speed=100, deg_per_sec=1e6, settle_s=0.0,
    )
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {SEARCHING}) == SEARCHING
    nsm.stop()


def test_inline_rescan_finds_target_after_timeout(monkeypatch):
    monkeypatch.setattr(sm, "RESCAN_AFTER_S", 0.05)

    bot = FakeBot()
    # No detection at all until the rescan's own scan pass finally sees it.
    pipeline = FakePipeline(detections_by_call=[[], [], [], [], [TARGET]])

    nsm = NavStateMachine(
        bot, pipeline, sonar=FakeSonar(distance_cm=200.0), skip_scan=True,
        n_steps=2, rotation_speed=100, deg_per_sec=1e6, settle_s=0.0,
    )
    nsm.start(target_label="bottle-plastic")

    assert _wait_for_state(nsm, {APPROACHING, FOUND}, timeout_s=5.0) in (APPROACHING, FOUND)
    nsm.stop()


def test_inline_rescan_without_target_returns_to_searching(monkeypatch):
    monkeypatch.setattr(sm, "RESCAN_AFTER_S", 0.05)

    bot = FakeBot()
    pipeline = FakePipeline()  # never any detections, ever

    nsm = NavStateMachine(
        bot, pipeline, sonar=FakeSonar(distance_cm=200.0), skip_scan=True,
        n_steps=2, rotation_speed=100, deg_per_sec=1e6, settle_s=0.0,
    )
    nsm.start(target_label="bottle-plastic")

    time.sleep(0.5)
    assert nsm.get_status()["state"] == SEARCHING
    nsm.stop()


def test_get_history_returns_sessions_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "LOG_DIR", tmp_path)

    (tmp_path / "20260101_000000.json").write_text(json.dumps({
        "session_id": "20260101_000000", "target": "a", "start_time": "t1",
        "duration_s": 1.0, "outcome": "FOUND", "detections_count": 1,
    }))
    (tmp_path / "20260102_000000.json").write_text(json.dumps({
        "session_id": "20260102_000000", "target": "b", "start_time": "t2",
        "duration_s": 2.0, "outcome": "STOPPED", "detections_count": 0,
    }))
    (tmp_path / "corrupt.json").write_text("{not valid json")

    bot = FakeBot()
    nsm = NavStateMachine(bot, FakePipeline(), sonar=None, skip_scan=True)

    history = nsm.get_history(limit=20)
    assert [h["session_id"] for h in history] == ["20260102_000000", "20260101_000000"]


def test_get_history_returns_empty_list_when_log_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "LOG_DIR", tmp_path / "does-not-exist")
    nsm = NavStateMachine(FakeBot(), FakePipeline(), sonar=None, skip_scan=True)
    assert nsm.get_history() == []


def test_finalize_session_survives_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "LOG_DIR", tmp_path)

    def _boom(self, *a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(sm.Path, "write_text", _boom)

    bot = FakeBot()
    pipeline = FakePipeline()
    nsm = NavStateMachine(bot, pipeline, sonar=None, skip_scan=True)
    nsm.start(target_label="bottle-plastic")

    time.sleep(0.3)
    nsm.stop()  # must not raise even though the session log couldn't be written
    assert nsm.get_status()["state"] == IDLE
