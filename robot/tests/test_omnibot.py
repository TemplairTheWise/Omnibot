"""Unit tests for OmniBot (omnibot.py).

OmniBot's constructor talks to real I2C/PCA9685/servo hardware, which would
physically move the robot's wheels if exercised for real. Every test here
patches busio.I2C, PCA9685 and servo.Servo BEFORE constructing OmniBot, so no
real hardware call is ever made and no motor ever actually turns.
"""

import math

import pytest

import omnibot as omnibot_module
from omnibot import OmniBot


@pytest.fixture
def bot(mocker):
    mocker.patch("omnibot.busio.I2C", return_value=mocker.Mock())
    mocker.patch("omnibot.PCA9685", return_value=mocker.Mock(channels=[mocker.Mock() for _ in range(8)]))
    mocker.patch("omnibot.servo.Servo", side_effect=lambda channel: mocker.Mock(angle=None))
    return OmniBot()


def _expected_angle(index: int, speed_percent: float) -> float:
    """Mirrors OmniBot.__setMotor()'s formula, referencing the real _TRIM constant."""
    trim = OmniBot._TRIM[index]
    trimmed = max(-100.0, min(100.0, speed_percent + trim))
    speed = (trimmed / 100) * 90
    if index % 2 == 0:
        speed = speed * -1
    return 90 + speed


def test_constructs_four_motors_without_touching_real_hardware(bot):
    assert len(bot.motors) == 4


def test_stop_sets_each_motor_to_its_trimmed_zero_angle(bot):
    bot.stop()
    for i in range(4):
        assert bot.motors[i].angle == pytest.approx(_expected_angle(i, 0.0), abs=1e-6)


def test_start_move_forward_drives_all_wheels_at_equal_speed(bot):
    bot.startMove([0.0, 1.0], 50.0)
    assert bot.motors[0].angle == pytest.approx(_expected_angle(0, 50.0), abs=1e-6)
    assert bot.motors[1].angle == pytest.approx(_expected_angle(1, 50.0), abs=1e-6)
    assert bot.motors[2].angle == pytest.approx(_expected_angle(2, 50.0), abs=1e-6)
    assert bot.motors[3].angle == pytest.approx(_expected_angle(3, 50.0), abs=1e-6)


def test_start_move_strafe_right_uses_differential_wheel_speeds(bot):
    bot.startMove([1.0, 0.0], 50.0)
    assert bot.motors[0].angle == pytest.approx(_expected_angle(0, 50.0), abs=1e-6)
    assert bot.motors[1].angle == pytest.approx(_expected_angle(1, -50.0), abs=1e-6)
    assert bot.motors[2].angle == pytest.approx(_expected_angle(2, -50.0), abs=1e-6)
    assert bot.motors[3].angle == pytest.approx(_expected_angle(3, 50.0), abs=1e-6)


def test_start_move_rejects_out_of_range_speed(bot):
    bot.startMove([0.0, 1.0], 150.0)
    for i in range(4):
        assert bot.motors[i].angle is None  # untouched - the call must be a no-op

    bot.startMove([0.0, 1.0], -150.0)
    for i in range(4):
        assert bot.motors[i].angle is None


def test_rotate_right_spins_left_and_right_wheels_oppositely(bot):
    bot.rotate("right", 50.0)
    assert bot.motors[0].angle == pytest.approx(_expected_angle(0, 50.0), abs=1e-6)
    assert bot.motors[2].angle == pytest.approx(_expected_angle(2, 50.0), abs=1e-6)
    assert bot.motors[1].angle == pytest.approx(_expected_angle(1, -50.0), abs=1e-6)
    assert bot.motors[3].angle == pytest.approx(_expected_angle(3, -50.0), abs=1e-6)


def test_rotate_left_is_the_mirror_of_rotate_right(bot):
    bot.rotate("left", 50.0)
    assert bot.motors[0].angle == pytest.approx(_expected_angle(0, -50.0), abs=1e-6)
    assert bot.motors[2].angle == pytest.approx(_expected_angle(2, -50.0), abs=1e-6)
    assert bot.motors[1].angle == pytest.approx(_expected_angle(1, 50.0), abs=1e-6)
    assert bot.motors[3].angle == pytest.approx(_expected_angle(3, 50.0), abs=1e-6)
