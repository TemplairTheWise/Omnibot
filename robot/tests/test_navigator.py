"""Unit tests for Navigator (navigator.py), using a FakeBot/FakeSonar - no hardware needed."""

import numpy as np
import pytest

from navigator import Navigator, SLOW_ZONE_CAP, ROTATION_TRIGGER, ROTATION_SPEED, BASE_SPEED, MIN_SPEED
from sonar_guard import ZONE_CLEAR, ZONE_SLOW, ZONE_STOP


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
    def __init__(self, zone=ZONE_CLEAR, distance_cm=200.0):
        self.zone = zone
        self.distance_cm = distance_cm


def _open_depth():
    return np.zeros((240, 320), dtype=np.float32)


def _blocked_depth():
    return np.full((240, 320), 0.9, dtype=np.float32)


def test_sonar_stop_zone_halts_without_driving():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar(zone=ZONE_STOP))
    result = nav.step(_open_depth(), goal_bearing_deg=0.0)
    assert result is None
    assert bot.calls == [("stop",)]


def test_check_sonar_false_ignores_stop_zone():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar(zone=ZONE_STOP))
    result = nav.step(_open_depth(), goal_bearing_deg=0.0, check_sonar=False)
    assert result is not None
    assert ("stop",) not in bot.calls
    assert any(call[0] == "startMove" for call in bot.calls)


def test_fully_blocked_scene_rotates_left_to_rescan():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar())
    result = nav.step(_blocked_depth(), goal_bearing_deg=0.0)
    assert result is None
    assert bot.calls == [("rotate", "left", ROTATION_SPEED)]


def test_open_scene_drives_forward_without_rotating():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar())
    result = nav.step(_open_depth(), goal_bearing_deg=0.0)
    assert result is not None
    assert bot.calls[0][0] == "startMove"


def test_large_steering_angle_rotates_in_place_instead_of_driving():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar())
    depth = _open_depth()
    depth[:, 100:] = 0.85  # wall covering most of the frame - only a narrow left gap remains
    result = nav.step(depth, goal_bearing_deg=25.0)  # goal points at the wall
    assert result is not None
    assert abs(result.angle_deg) > ROTATION_TRIGGER
    assert bot.calls[0][0] == "rotate"
    assert bot.calls[0][1] == "left"  # steering left, away from the wall


def test_speed_capped_in_slow_sonar_zone():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar(zone=ZONE_SLOW, distance_cm=40.0))
    nav.step(_open_depth(), goal_bearing_deg=0.0)
    assert bot.calls[0][0] == "startMove"
    speed = bot.calls[0][2]
    assert speed <= SLOW_ZONE_CAP


def test_speed_scales_with_clearance_between_min_and_base():
    bot = FakeBot()
    nav = Navigator(bot, sonar=FakeSonar())
    depth = _open_depth()
    depth[:, 160:] = 0.85  # right half blocked -> partial clearance
    nav.step(depth, goal_bearing_deg=0.0)
    drive_calls = [c for c in bot.calls if c[0] == "startMove"]
    if drive_calls:
        speed = drive_calls[0][2]
        assert MIN_SPEED <= speed <= BASE_SPEED
