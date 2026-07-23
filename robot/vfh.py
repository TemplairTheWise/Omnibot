"""
VFH (Vector Field Histogram) steering module — Step 2.

Converts a depth map from the inference pipeline into a steering angle
that avoids obstacles while staying biased toward a goal bearing.

Public API:
    vfh = VFH()
    angle = vfh.compute(depth_map, goal_bearing_deg=0.0)
    if angle is None:
        # every sector is blocked — caller should rotate to rescan
    vx, vy = vfh.angle_to_vector(angle)
    bot.startMove([vx, vy], speed)

Coordinate conventions
----------------------
  angle = 0   → straight ahead
  angle > 0   → right of forward axis
  angle < 0   → left of forward axis

  (vx, vy) maps directly to OmniBot.startMove([vx, vy], speed):
    vx = sin(angle), vy = cos(angle)

Depth map contract (from inference_pipeline._parse_depth)
---------------------------------------------------------
  shape  : (H, W) float32
  values : per-frame min-max normalised → 0 = furthest, 1 = nearest
  Because the scale is relative, safe_threshold is also relative:
  a sector is "blocked" when its worst-case depth > safe_threshold,
  meaning it contains one of the closer objects in the current scene.
  The HC-SR04 thread provides the absolute collision hard-stop.
"""

import math
from typing import NamedTuple

import numpy as np

# ── Tunables ──────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "n_sectors":         36,   # angular bins covering the camera FOV
    "camera_fov_deg":    62.0, # RPi Camera Module 3 horizontal FOV
    "safe_threshold":    0.65, # clearance below this → sector blocked (0=near, 1=far)
    "smoothing_sigma":   1.5,  # Gaussian σ in sector units
    "goal_bias":         0.6,  # 0 = pure avoidance, 1 = steer directly to goal
    "min_valley_width":  2,    # sectors — narrower valleys are ignored
    "ground_crop":       0.30, # fraction of image height to discard from the bottom
                               # Removes the floor (always the nearest surface after
                               # per-frame normalisation) so wall depth values are not
                               # compressed relative to it.  Tune up if floor still
                               # bleeds into clearance; down if the camera is tilted up.
}


class VFHResult(NamedTuple):
    angle_deg:   float              # chosen steering angle (degrees from forward)
    vx:          float              # lateral component for startMove
    vy:          float              # forward component for startMove
    clearance:   np.ndarray         # smoothed clearance histogram (N,) for debug
    valley_mask: np.ndarray         # boolean mask of passable sectors (N,)


# ── VFH class ─────────────────────────────────────────────────────────────────
class VFH:
    """
    Stateless steering computer — safe to call from any thread.
    All tunable parameters can be overridden at construction time.
    """

    def __init__(self, **kwargs):
        cfg = {**_DEFAULTS, **kwargs}
        self.n_sectors        = int(cfg["n_sectors"])
        self.camera_fov_deg   = float(cfg["camera_fov_deg"])
        self.safe_threshold   = float(cfg["safe_threshold"])
        self.smoothing_sigma  = float(cfg["smoothing_sigma"])
        self.goal_bias        = float(cfg["goal_bias"])
        self.min_valley_width = int(cfg["min_valley_width"])
        self.ground_crop      = float(cfg["ground_crop"])

    # ── Public interface ──────────────────────────────────────────────────────

    def compute(
        self,
        depth_map: np.ndarray,
        goal_bearing_deg: float = 0.0,
    ) -> VFHResult | None:
        """
        Compute the best steering direction.

        Parameters
        ----------
        depth_map        : (H, W) float32, 0=far 1=near from inference_pipeline
        goal_bearing_deg : desired heading (degrees), 0 = straight ahead

        Returns
        -------
        VFHResult   if a passable valley exists
        None        if all sectors are blocked (caller should rotate to rescan)
        """
        clearance    = self._build_clearance(depth_map)
        valley_mask  = clearance > self.safe_threshold
        valleys      = self._find_valleys(valley_mask)

        if not valleys:
            return None

        goal_sector  = self._angle_to_sector(goal_bearing_deg)
        best         = self._score_valleys(valleys, clearance, goal_sector)
        angle_deg    = self._valley_steer_angle(best, goal_sector)
        vx, vy       = self.angle_to_vector(angle_deg)

        return VFHResult(angle_deg, vx, vy, clearance, valley_mask)

    @staticmethod
    def angle_to_vector(angle_deg: float) -> tuple[float, float]:
        """
        Steering angle → (vx, vy) unit move vector.
        vx = lateral (+ right), vy = longitudinal (+ forward).
        """
        rad = math.radians(angle_deg)
        return math.sin(rad), math.cos(rad)

    def sector_angle_deg(self, sector_idx: float) -> float:
        """Sector index → bearing in degrees (0 = forward, + = right)."""
        return (sector_idx / self.n_sectors - 0.5) * self.camera_fov_deg

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _build_clearance(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Slice the depth image into N_SECTORS vertical columns.
        Take the max depth per column (worst-case obstacle in that slice),
        invert to get clearance, then smooth.

        Returns clearance[N] in [0, 1]: 1 = completely clear, 0 = fully blocked.
        """
        # Drop the bottom fraction of the image to exclude the floor.
        # The floor is always the nearest surface in the frame, so without
        # this crop it dominates per-frame normalisation and makes walls at
        # moderate distances look shallower (less dangerous) than they are.
        if self.ground_crop > 0:
            keep_rows = int(depth_map.shape[0] * (1.0 - self.ground_crop))
            depth_map = depth_map[:keep_rows, :]

        cols = np.array_split(depth_map, self.n_sectors, axis=1)
        worst = np.array([col.max() for col in cols], dtype=np.float32)
        clearance = 1.0 - worst
        return self._gaussian_smooth(clearance, self.smoothing_sigma)

    def _angle_to_sector(self, angle_deg: float) -> float:
        """Bearing in degrees → (possibly fractional) sector index."""
        return (angle_deg / self.camera_fov_deg + 0.5) * self.n_sectors

    def _find_valleys(self, passable: np.ndarray) -> list[dict]:
        """
        Find contiguous runs of passable (True) sectors.
        Each valley is {"start": int, "end": int} (inclusive).
        """
        valleys: list[dict] = []
        in_v, start = False, 0
        for i, p in enumerate(passable):
            if p and not in_v:
                start, in_v = i, True
            elif not p and in_v:
                if (i - start) >= self.min_valley_width:
                    valleys.append({"start": start, "end": i - 1})
                in_v = False
        if in_v and (len(passable) - start) >= self.min_valley_width:
            valleys.append({"start": start, "end": len(passable) - 1})
        return valleys

    def _score_valleys(
        self,
        valleys: list[dict],
        clearance: np.ndarray,
        goal_sector: float,
    ) -> dict:
        """
        Score each valley and return the best one.

        score = mean_clearance × width  −  goal_bias × angular_distance_to_goal
        The angular distance is in sector units (not degrees), keeping both
        terms on a comparable scale.
        """
        best_score, best = float("-inf"), valleys[0]
        for v in valleys:
            width = v["end"] - v["start"] + 1
            center = (v["start"] + v["end"]) / 2.0
            mean_cl = float(clearance[v["start"]: v["end"] + 1].mean())
            goal_dist = abs(center - goal_sector)
            score = mean_cl * width - self.goal_bias * goal_dist
            if score > best_score:
                best_score, best = score, v
        return best

    def _valley_steer_angle(self, valley: dict, goal_sector: float) -> float:
        """
        Pick the steering target within the selected valley.

        If the goal sector falls inside the valley, steer directly toward it.
        Otherwise steer to the valley centre.
        """
        s, e = valley["start"], valley["end"]
        if s <= goal_sector <= e:
            target_sector = goal_sector
        else:
            target_sector = (s + e) / 2.0
        return self.sector_angle_deg(target_sector)

    @staticmethod
    def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
        """1-D Gaussian convolution with edge-padding (no wrap-around)."""
        radius = max(1, round(3 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(arr, radius, mode="edge")
        return np.convolve(padded, kernel, mode="valid")


# ── ASCII debug visualiser ────────────────────────────────────────────────────
def debug_histogram(result: VFHResult, width: int = 72) -> str:
    """
    Return a multiline ASCII visualisation of the clearance histogram.
    Blocked sectors shown as '#', clear sectors as their clearance bar,
    chosen direction marked with '^'.
    """
    n = len(result.clearance)
    bar_w = max(1, width // n)
    rows = 8
    lines = []
    for row in range(rows, -1, -1):
        thresh = row / rows
        line = ""
        for i, cl in enumerate(result.clearance):
            if not result.valley_mask[i]:
                ch = "#" * bar_w          # blocked sector
            elif cl >= thresh:
                ch = "█" * bar_w
            else:
                ch = " " * bar_w
            line += ch
        lines.append(line)

    # Direction marker
    fov = len(result.clearance)
    marker_pos = int((result.angle_deg / 62.0 + 0.5) * fov * bar_w)
    marker_pos = max(0, min(width - 1, marker_pos))
    lines.append(" " * marker_pos + "^")
    lines.append(f"  angle={result.angle_deg:+.1f}°  vx={result.vx:+.2f}  vy={result.vy:.2f}")
    return "\n".join(lines)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vfh = VFH()

    print("=" * 60)
    print("Test 1: wall on the right, goal straight ahead → steer left-of-centre")
    depth = np.zeros((256, 320), dtype=np.float32)
    depth[:, 200:] = 0.85   # obstacle occupying right ~37% of frame
    result = vfh.compute(depth, goal_bearing_deg=0.0)
    if result:
        print(debug_histogram(result))
    else:
        print("  ALL BLOCKED")

    print()
    print("=" * 60)
    print("Test 2: narrow gap on the left, goal on the right → steer through gap")
    depth2 = np.full((256, 320), 0.85, dtype=np.float32)
    depth2[:, :80] = 0.1   # only the left ~25% is clear
    result2 = vfh.compute(depth2, goal_bearing_deg=20.0)
    if result2:
        print(debug_histogram(result2))
    else:
        print("  ALL BLOCKED")

    print()
    print("=" * 60)
    print("Test 3: fully blocked → returns None")
    depth3 = np.full((256, 320), 0.9, dtype=np.float32)
    result3 = vfh.compute(depth3)
    print("  Result:", result3)

    print()
    print("=" * 60)
    print("Test 4: open scene, goal 15° right → steers toward goal")
    depth4 = np.full((256, 320), 0.05, dtype=np.float32)
    result4 = vfh.compute(depth4, goal_bearing_deg=15.0)
    if result4:
        print(debug_histogram(result4))
