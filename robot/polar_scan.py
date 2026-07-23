"""
polar_scan.py — Step 6: 360° startup scan.

Rotates the robot through a full circle in equal-time increments, sampling
depth and detections at each step to produce:

    clearance_map       — mean VFH clearance for each angular bin
    best_heading_deg    — the most open direction (highest clearance)
    target_heading_deg  — global bearing to the first detected target, or None

There is no odometry; headings are estimated from timed rotation.
DEG_PER_SEC must be calibrated for your robot at the chosen ROTATION_SPEED.

Calibration procedure
---------------------
1. Mark the robot's starting orientation on the floor.
2. Run: python force_test.py  (or call bot.rotate("right", ROTATION_SPEED) for 2 s)
3. Measure the actual angle turned.
4. Set DEG_PER_SEC = measured_angle_deg / 2.0 in this file.

Public API
----------
    scanner = PolarScan(bot, pipeline)
    result  = scanner.run(target_label="bottle-plastic")

    result.target_heading_deg   # heading to target (°), or None if not seen
    result.best_heading_deg     # most open direction (°)
    result.clearance_map        # (n_steps,) float32 — clearance per heading
    result.completed            # False if scan aborted early (target found)

    scanner.face_heading(result.best_heading_deg)   # rotate to face that direction

Standalone test (requires hardware)
------------------------------------
    python polar_scan.py [--target bottle-plastic] [--no-move]
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from omnibot import OmniBot
from vfh import VFH

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
N_STEPS        = 18     # angular bins — 360 / 18 = 20° per step
ROTATION_SPEED = 100    # % — calibrated measurement was taken at full speed
DEG_PER_SEC    = 45.0   # °/s at ROTATION_SPEED=100 — measured: 80-100° in 2 s → 45°/s mid
SETTLE_S       = 0.5    # seconds to pause after each step (camera + depth model settle)
ROTATE_DIR     = "right"  # scan direction

STEP_ANGLE_DEG = 360.0 / N_STEPS   # 20°


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    clearance_map:      np.ndarray        # (n_steps,) mean VFH clearance per heading
    target_heading_deg: float | None      # global bearing to detected target, or None
    best_heading_deg:   float             # most open global heading
    target_label:       str | None        # what was searched for (None = any)
    n_steps:            int
    step_angle_deg:     float
    completed:          bool              # False if scan aborted early when target found


# ── PolarScan class ───────────────────────────────────────────────────────────

class PolarScan:
    """
    Executes a timed 360° rotation, sampling depth + detections at each step.
    """

    def __init__(
        self,
        bot:            OmniBot,
        pipeline,                         # InferencePipeline
        n_steps:        int   = N_STEPS,
        rotation_speed: int   = ROTATION_SPEED,
        deg_per_sec:    float = DEG_PER_SEC,
        settle_s:       float = SETTLE_S,
        camera_fov_deg: float = 62.0,
        stop_on_target: bool  = True,    # abort as soon as target first seen
        **vfh_kwargs,
    ):
        self.bot            = bot
        self.pipeline       = pipeline
        self.n_steps        = n_steps
        self.rotation_speed = rotation_speed
        self.deg_per_sec    = deg_per_sec
        self.step_angle_deg = 360.0 / n_steps
        self.step_dur       = self.step_angle_deg / deg_per_sec
        self.settle_s       = settle_s
        self.camera_fov_deg = camera_fov_deg
        self.stop_on_target = stop_on_target
        self.vfh            = VFH(**vfh_kwargs)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, target_label: str | None = None) -> ScanResult:
        """
        Rotate through 360°, sampling depth + detections at each step.

        If target_label is given and stop_on_target is True, the scan stops
        as soon as the target is detected; the robot is left facing the target.

        All returned headings are relative to the robot's orientation at the
        moment run() is called (0° = initial forward direction, + = clockwise).
        """
        clearance_map  = np.zeros(self.n_steps, dtype=np.float32)
        target_heading = None
        completed      = True

        total_deg = self.n_steps * self.step_angle_deg
        est_s     = self.n_steps * (self.step_dur + self.settle_s)
        log.info(
            "PolarScan: %d steps × %.0f° = %.0f°  "
            "step_dur=%.2fs  settle=%.2fs  est_total=%.0fs  target=%s",
            self.n_steps, self.step_angle_deg, total_deg,
            self.step_dur, self.settle_s, est_s,
            target_label or "any",
        )

        for step in range(self.n_steps):
            heading = step * self.step_angle_deg

            # ── Rotate one increment ─────────────────────────────────────────
            self.bot.rotate(ROTATE_DIR, self.rotation_speed)
            time.sleep(self.step_dur)
            self.bot.stop()
            time.sleep(self.settle_s)

            # ── Sample ───────────────────────────────────────────────────────
            _, depth, detections = self.pipeline.get_state()

            if depth is not None:
                vfh_result = self.vfh.compute(depth, goal_bearing_deg=0.0)
                if vfh_result is not None:
                    clearance_map[step] = float(vfh_result.clearance.mean())

            log.debug(
                "Step %2d/%d  heading=%5.1f°  clearance=%.2f  dets=%d",
                step + 1, self.n_steps, heading,
                clearance_map[step], len(detections),
            )

            # ── Check for target ─────────────────────────────────────────────
            if target_heading is None:
                candidates = [
                    d for d in detections
                    if target_label is None or d.label == target_label
                ]
                if candidates:
                    best = max(candidates,
                               key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
                    cx            = (best.x1 + best.x2) / 2.0
                    frame_bearing = (cx - 0.5) * self.camera_fov_deg
                    target_heading = (heading + frame_bearing) % 360.0
                    log.info(
                        "Target '%s' found at step %d — "
                        "frame_bearing=%+.1f°  global_heading=%.1f°",
                        best.label, step + 1, frame_bearing, target_heading,
                    )
                    if self.stop_on_target:
                        completed = False
                        break

        self.bot.stop()

        best_step    = int(np.argmax(clearance_map))
        best_heading = best_step * self.step_angle_deg

        log.info(
            "PolarScan complete — best=%.0f°  target=%s  completed=%s",
            best_heading,
            f"{target_heading:.1f}°" if target_heading is not None else "none",
            completed,
        )

        return ScanResult(
            clearance_map      = clearance_map,
            target_heading_deg = target_heading,
            best_heading_deg   = best_heading,
            target_label       = target_label,
            n_steps            = self.n_steps,
            step_angle_deg     = self.step_angle_deg,
            completed          = completed,
        )

    def face_heading(self, heading_deg: float) -> None:
        """
        Rotate the robot to face a global heading from the scan reference frame.

        Chooses the shorter arc (≤ 180°) to minimise turn time.
        Call this after run() returns to align the robot before driving.

        Note: if the scan was aborted early (completed=False), the robot is
        already facing roughly the right way — you may not need to call this.
        """
        angle = heading_deg % 360.0

        if angle > 180.0:
            direction = "left" if ROTATE_DIR == "right" else "right"
            angle = 360.0 - angle
        else:
            direction = ROTATE_DIR

        if angle < 2.0:
            log.debug("face_heading: already facing target direction")
            return

        duration = angle / self.deg_per_sec
        log.info("face_heading: rotating %s %.1f° (%.2fs)", direction, angle, duration)
        self.bot.rotate(direction, self.rotation_speed)
        time.sleep(duration)
        self.bot.stop()
        time.sleep(self.settle_s)


# ── ASCII visualiser ──────────────────────────────────────────────────────────

def _ascii_polar(result: ScanResult, width: int = 60) -> str:
    """Print a horizontal bar chart of the clearance map."""
    lines = ["", f"  Clearance map ({result.n_steps} steps × {result.step_angle_deg:.0f}°)",
             "  " + "─" * width]
    for i, cl in enumerate(result.clearance_map):
        heading = i * result.step_angle_deg
        bar_len = int(cl * (width - 20))
        bar     = "█" * bar_len
        markers = ""
        if result.target_heading_deg is not None:
            if abs(heading - result.target_heading_deg) < result.step_angle_deg / 2:
                markers += " ◀ TARGET"
        if abs(heading - result.best_heading_deg) < result.step_angle_deg / 2:
            markers += " ◀ BEST"
        lines.append(f"  {heading:5.0f}°  {bar:<{width - 20}s}  {cl:.2f}{markers}")
    lines.append("  " + "─" * width)
    return "\n".join(lines)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from inference_pipeline import InferencePipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="360° polar scan test")
    parser.add_argument("--target",    type=str,   default=None,
                        help="beverage label to search for")
    parser.add_argument("--no-move",   action="store_true",
                        help="dry run: skip bot commands (camera + inference only)")
    parser.add_argument("--steps",     type=int,   default=N_STEPS,
                        help=f"number of scan steps (default: {N_STEPS})")
    parser.add_argument("--speed",     type=int,   default=ROTATION_SPEED,
                        help=f"rotation speed %% (default: {ROTATION_SPEED})")
    parser.add_argument("--deg-per-sec", type=float, default=DEG_PER_SEC,
                        help=f"calibrated angular velocity °/s (default: {DEG_PER_SEC})")
    args = parser.parse_args()

    pipeline = InferencePipeline()
    pipeline.start()

    log.info("Waiting for first depth frame …")
    while True:
        _, depth, _ = pipeline.get_state()
        if depth is not None:
            break
        time.sleep(0.1)
    log.info("Ready.")

    bot = OmniBot()

    class _NullBot:
        """Drop-in for OmniBot that skips all motor commands."""
        def rotate(self, *a, **kw): pass
        def stop(self, *a, **kw):   pass

    scanner = PolarScan(
        _NullBot() if args.no_move else bot,
        pipeline,
        n_steps        = args.steps,
        rotation_speed = args.speed,
        deg_per_sec    = args.deg_per_sec,
    )

    try:
        result = scanner.run(target_label=args.target)
        print(_ascii_polar(result))
        print(f"\n  Best heading : {result.best_heading_deg:.0f}°")
        if result.target_heading_deg is not None:
            print(f"  Target heading: {result.target_heading_deg:.1f}°")
        else:
            print("  Target        : not detected")
    except KeyboardInterrupt:
        bot.stop()
        log.info("Scan interrupted.")
    finally:
        pipeline.stop()
