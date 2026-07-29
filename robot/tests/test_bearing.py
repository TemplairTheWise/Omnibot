"""Unit tests for goal_bearing_from_detections() (inference_pipeline.py). Pure logic - no hardware."""

import pytest

from inference_pipeline import Detection, goal_bearing_from_detections


def _det(label, confidence, x1, y1, x2, y2):
    return Detection(label=label, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)


def test_no_detections_returns_none():
    assert goal_bearing_from_detections([], target_label=None) is None


def test_no_matching_target_returns_none():
    dets = [_det("tin can", 0.9, 0.4, 0.4, 0.6, 0.6)]
    assert goal_bearing_from_detections(dets, target_label="bottle-plastic") is None


def test_centered_detection_has_near_zero_bearing():
    dets = [_det("bottle-plastic", 0.9, 0.4, 0.3, 0.6, 0.9)]
    bearing = goal_bearing_from_detections(dets, target_label="bottle-plastic", camera_fov_deg=62.0)
    assert bearing == pytest.approx(0.0, abs=1e-6)


def test_detection_on_right_has_positive_bearing():
    dets = [_det("bottle-plastic", 0.9, 0.8, 0.3, 1.0, 0.9)]
    bearing = goal_bearing_from_detections(dets, target_label="bottle-plastic", camera_fov_deg=62.0)
    assert bearing > 0


def test_detection_on_left_has_negative_bearing():
    dets = [_det("bottle-plastic", 0.9, 0.0, 0.3, 0.2, 0.9)]
    bearing = goal_bearing_from_detections(dets, target_label="bottle-plastic", camera_fov_deg=62.0)
    assert bearing < 0


def test_selects_largest_box_when_target_label_is_none():
    small = _det("tin can", 0.99, 0.0, 0.0, 0.1, 0.1)          # tiny box, far left
    big   = _det("bottle-plastic", 0.5, 0.45, 0.4, 0.55, 0.9)  # big box, centered
    bearing = goal_bearing_from_detections([small, big], target_label=None, camera_fov_deg=62.0)
    assert bearing == pytest.approx(0.0, abs=1.0)


def test_target_label_filters_out_bigger_box_of_other_class():
    small_match = _det("bottle-plastic", 0.9, 0.8, 0.3, 1.0, 0.9)  # smaller, right side, matches target
    big_other   = _det("tin can", 0.9, 0.0, 0.0, 1.0, 1.0)         # huge box, but wrong label
    bearing = goal_bearing_from_detections(
        [small_match, big_other], target_label="bottle-plastic", camera_fov_deg=62.0
    )
    assert bearing > 0  # must pick the smaller, correctly-labelled detection on the right
