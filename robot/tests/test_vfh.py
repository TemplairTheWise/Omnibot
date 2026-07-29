"""Unit tests for the VFH steering algorithm (vfh.py). Pure numpy logic - no hardware."""

import numpy as np
import pytest

from vfh import VFH


def test_open_scene_steers_toward_goal():
    vfh = VFH()
    depth = np.zeros((240, 320), dtype=np.float32)  # everything far -> fully clear
    result = vfh.compute(depth, goal_bearing_deg=0.0)
    assert result is not None
    assert abs(result.angle_deg) < 5.0


def test_fully_blocked_scene_returns_none():
    vfh = VFH()
    depth = np.full((240, 320), 0.9, dtype=np.float32)  # everything near -> blocked
    result = vfh.compute(depth, goal_bearing_deg=0.0)
    assert result is None


def test_steers_into_open_valley_when_goal_is_blocked():
    vfh = VFH()
    depth = np.zeros((240, 320), dtype=np.float32)
    depth[:, 200:] = 0.85  # obstacle occupying the right ~37% of the frame
    # Goal points straight at the blocked area on the right - the algorithm must
    # steer into the open valley on the left instead of driving into the wall.
    result = vfh.compute(depth, goal_bearing_deg=25.0)
    assert result is not None
    assert result.angle_deg < 0


def test_prefers_narrow_gap_over_blocked_goal_direction():
    vfh = VFH()
    depth = np.full((240, 320), 0.85, dtype=np.float32)
    depth[:, :80] = 0.1  # only the left ~25% of the frame is clear
    result = vfh.compute(depth, goal_bearing_deg=20.0)  # goal is to the right, but blocked
    assert result is not None
    assert result.angle_deg < 0  # steers through the gap on the left instead of toward goal


def test_angle_to_vector_forward_and_right():
    vx, vy = VFH.angle_to_vector(0.0)
    assert vx == pytest.approx(0.0, abs=1e-6)
    assert vy == pytest.approx(1.0, abs=1e-6)

    vx, vy = VFH.angle_to_vector(90.0)
    assert vx == pytest.approx(1.0, abs=1e-6)
    assert vy == pytest.approx(0.0, abs=1e-6)


def test_sector_angle_deg_center_sector_is_zero():
    vfh = VFH(n_sectors=36, camera_fov_deg=62.0)
    center = vfh.n_sectors / 2
    assert vfh.sector_angle_deg(center) == pytest.approx(0.0, abs=1e-6)


def test_ground_crop_excludes_floor_from_clearance():
    vfh = VFH(ground_crop=0.30, n_sectors=8, min_valley_width=1)
    depth = np.zeros((100, 80), dtype=np.float32)
    depth[70:, :] = 1.0  # bottom 30% is "floor" - nearest surface after normalisation
    result = vfh.compute(depth, goal_bearing_deg=0.0)
    # With the floor cropped out of the clearance computation, the scene should
    # still read as open rather than being dominated by the floor.
    assert result is not None
    assert abs(result.angle_deg) < 10.0
