"""Unit tests for robot_server.py (Flask API + web UI).

robot_server.py wires up real hardware (OmniBot, InferencePipeline, SonarGuard) at
MODULE IMPORT TIME. To test its Flask routes without ever touching real motors,
the real camera, or the real Hailo device, OmniBot / InferencePipeline / SonarGuard
are replaced with lightweight fakes *before* robot_server is imported for the
first time, using sys.modules eviction + a fresh import.
"""

import importlib
import sys
import time
from unittest import mock

import pytest


class _FakeOmniBot:
    def __init__(self):
        self.calls = []
        self.pca = mock.MagicMock(channels=[mock.MagicMock() for _ in range(8)])

    def startMove(self, vect, speed):
        self.calls.append(("startMove", vect, speed))

    def rotate(self, direction, speed):
        self.calls.append(("rotate", direction, speed))

    def stop(self):
        self.calls.append(("stop",))


class _FakeInferencePipeline:
    def __init__(self):
        self.recording = None

    def start(self):
        pass

    def stop(self):
        pass

    def get_state(self):
        return None, None, []

    def start_recording(self, path, flip=False):
        self.recording = (path, flip)

    def stop_recording(self):
        self.recording = None


class _FakeSonarGuard:
    def __init__(self):
        self.distance_cm = None
        self.zone = "clear"

    def start(self):
        pass

    def stop(self):
        pass


def _import_robot_server_with_working_navigation():
    import omnibot, inference_pipeline, sonar_guard
    import adafruit_motor.servo as servo_module

    mock.patch.object(omnibot, "OmniBot", _FakeOmniBot).start()
    mock.patch.object(inference_pipeline, "InferencePipeline", _FakeInferencePipeline).start()
    mock.patch.object(sonar_guard, "SonarGuard", _FakeSonarGuard).start()
    mock.patch.object(servo_module, "Servo", lambda channel: mock.MagicMock(angle=None)).start()

    sys.modules.pop("robot_server", None)
    return importlib.import_module("robot_server")


@pytest.fixture(scope="module")
def app_module():
    module = _import_robot_server_with_working_navigation()
    yield module
    sys.modules.pop("robot_server", None)
    mock.patch.stopall()


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture(autouse=True)
def _reset_nsm(app_module):
    """Every test starts from a clean IDLE state, regardless of test order."""
    app_module._nsm.stop()
    app_module.bot.calls.clear()
    yield
    app_module._nsm.stop()


def test_navigation_wired_up_with_fakes(app_module):
    assert app_module._nsm is not None
    assert isinstance(app_module.bot, _FakeOmniBot)


def test_move_drives_the_bot(client, app_module):
    resp = client.post("/move", json={"x": 0.0, "y": 1.0, "speed": 50})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    assert ("startMove", [0.0, 1.0], 50.0) in app_module.bot.calls


def test_move_clamps_speed_to_valid_range(client, app_module):
    client.post("/move", json={"x": 0.0, "y": 1.0, "speed": 500})
    call = next(c for c in app_module.bot.calls if c[0] == "startMove")
    assert call[2] == 100.0


def test_rotate_drives_the_bot(client, app_module):
    resp = client.post("/rotate", json={"direction": "left", "speed": 40})
    assert resp.status_code == 200
    assert ("rotate", "left", 40.0) in app_module.bot.calls


def test_stop_halts_the_bot(client, app_module):
    resp = client.post("/stop", json={})
    assert resp.status_code == 200
    assert ("stop",) in app_module.bot.calls


def test_gripper_toggle_flips_state(client, app_module):
    resp1 = client.post("/gripper", json={"toggle": True})
    closed1 = resp1.get_json()["closed"]
    resp2 = client.post("/gripper", json={"toggle": True})
    closed2 = resp2.get_json()["closed"]
    assert closed1 != closed2


def test_gripper_open_field_sets_state_explicitly(client, app_module):
    resp = client.post("/gripper", json={"open": False})
    assert resp.get_json()["closed"] is True
    resp2 = client.post("/gripper", json={"open": True})
    assert resp2.get_json()["closed"] is False


def test_status_reports_gripper_state(client, app_module):
    client.post("/gripper", json={"open": True})
    resp = client.get("/status")
    assert resp.get_json() == {"gripper_closed": False}


def test_search_labels_returns_beverage_labels(client):
    resp = client.get("/search/labels")
    labels = resp.get_json()
    assert "bottle-plastic" in labels
    assert len(labels) == 9


def test_search_status_reports_idle_before_any_search(client):
    resp = client.get("/search/status")
    assert resp.get_json()["state"] == "IDLE"


def test_search_start_and_stop_roundtrip(client, app_module):
    resp = client.post("/search/start", json={"target": "bottle-plastic"})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True

    resp2 = client.post("/search/start", json={"target": "bottle-plastic"})
    assert resp2.status_code == 409  # already running

    resp3 = client.post("/search/stop", json={})
    assert resp3.status_code == 200
    time.sleep(0.2)
    assert app_module._nsm.get_status()["state"] == "IDLE"


def test_move_stops_an_active_search_first(client, app_module):
    client.post("/search/start", json={"target": "bottle-plastic"})
    time.sleep(0.1)
    client.post("/move", json={"x": 1.0, "y": 0.0, "speed": 30})
    time.sleep(0.1)
    assert app_module._nsm.get_status()["state"] == "IDLE"


def test_search_history_is_a_list(client):
    resp = client.get("/search/history")
    assert isinstance(resp.get_json(), list)


def test_index_page_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"OmniBot" in resp.data


# ── Degraded mode: navigation hardware unavailable at import time ────────────

@pytest.fixture(scope="module")
def app_module_no_nav():
    import omnibot
    import adafruit_motor.servo as servo_module
    import state_machine

    mock.patch.object(omnibot, "OmniBot", _FakeOmniBot).start()
    mock.patch.object(servo_module, "Servo", lambda channel: mock.MagicMock(angle=None)).start()
    # Simulate the Hailo stack failing to initialise (e.g. no accelerator attached).
    mock.patch.object(
        state_machine, "NavStateMachine", side_effect=RuntimeError("no Hailo device")
    ).start()

    sys.modules.pop("robot_server", None)
    module = importlib.import_module("robot_server")
    yield module
    sys.modules.pop("robot_server", None)
    mock.patch.stopall()


def test_server_still_starts_when_navigation_hardware_is_absent(app_module_no_nav):
    assert app_module_no_nav._nsm is None


def test_manual_control_still_works_without_navigation(app_module_no_nav):
    client = app_module_no_nav.app.test_client()
    resp = client.post("/move", json={"x": 0.0, "y": 1.0, "speed": 50})
    assert resp.status_code == 200


def test_search_endpoints_report_unavailable_without_navigation(app_module_no_nav):
    client = app_module_no_nav.app.test_client()
    assert client.post("/search/start", json={}).status_code == 503
    assert client.post("/search/stop", json={}).status_code == 503
    assert client.get("/search/status").get_json()["state"] == "unavailable"
    assert client.get("/search/history").get_json() == []
