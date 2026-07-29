"""Unit tests for SonarGuard (sonar_guard.py) using gpiozero's MockFactory - no real HC-SR04 needed."""

import time

import pytest
from gpiozero import Device
from gpiozero.pins.mock import MockFactory

from sonar_guard import SonarGuard, ZONE_CLEAR, ZONE_STOP


class _FakeSensor:
    """Drop-in replacement for gpiozero.DistanceSensor exposing just `.distance` (metres)."""

    def __init__(self, distance_m: float):
        self.distance = distance_m

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _mock_gpio():
    Device.pin_factory = MockFactory()
    yield
    Device.pin_factory.reset()


def _make_guard(poll_hz=50):
    guard = SonarGuard(poll_hz=poll_hz, stop_count=3)
    guard._sensor = _FakeSensor(2.0)  # start far away (clear)
    return guard


def test_clear_zone_when_far():
    guard = _make_guard()
    guard.start()
    try:
        time.sleep(0.3)
        assert guard.zone == ZONE_CLEAR
        assert guard.distance_cm == pytest.approx(200.0, abs=5.0)
    finally:
        guard.stop()


def test_stop_zone_after_sustained_close_readings():
    guard = _make_guard()
    guard.start()
    try:
        time.sleep(0.15)
        assert guard.zone == ZONE_CLEAR
        guard._sensor.distance = 0.05  # 5 cm - well inside the STOP zone
        time.sleep(0.3)
        assert guard.zone == ZONE_STOP
    finally:
        guard.stop()


def test_spurious_near_zero_readings_are_filtered():
    guard = _make_guard()
    guard.start()
    try:
        time.sleep(0.15)
        guard._sensor.distance = 0.01  # 1 cm - below MIN_VALID_CM, should be discarded
        time.sleep(0.3)
        assert guard.zone == ZONE_CLEAR
    finally:
        guard.stop()
