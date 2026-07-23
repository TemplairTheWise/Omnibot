"""
navigator.py — Step 3: VFH steering → OmniBot live commands.

Translates depth maps into startMove / rotate calls on the OmniBot.
Speed is modulated by the mean clearance of passable sectors:
  open space  → BASE_SPEED
  tight space → MIN_SPEED

The HC-SR04 hard-stop is wired in Step 4 and will sit above this layer.

Public API
----------
    nav = Navigator(bot)
    result = nav.step(depth_map, goal_bearing_deg=0.0)
    # result is a VFHResult (or None if all sectors were blocked)

Standalone test (pure VFH obstacle avoidance, no goal target):
    python navigator.py
"""

import logging
import time

import numpy as np

from omnibot import OmniBot
from sonar_guard import ZONE_SLOW, ZONE_STOP, SonarGuard
from vfh import VFH, VFHResult, debug_histogram

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
BASE_SPEED        = 80    # % — used when path is fully open
MIN_SPEED         = 35    # % — floor when most sectors are blocked
ROTATION_SPEED    = 35    # % — spot-rotation speed
SLOW_ZONE_CAP     = 45    # % — max speed while in sonar SLOW zone
ROTATION_TRIGGER  = 20.0  # ° — rotate in place above this steering angle
LOOP_HZ           = 10    # target update rate for the standalone test loop


class Navigator:
    """
    Single-frame steering controller.

    step() is designed to be called on every inference frame by the
    state machine (Step 7).  The standalone run() method drives the
    robot using pure VFH obstacle avoidance for manual testing.
    """

    def __init__(self, bot: OmniBot, sonar: SonarGuard | None = None, **vfh_kwargs):
        """
        Parameters
        ----------
        bot         : initialised OmniBot instance
        sonar       : optional SonarGuard; if provided, step() enforces distance zones
        **vfh_kwargs: forwarded to VFH() — override n_sectors, safe_threshold, etc.
        """
        self.bot              = bot
        self.sonar            = sonar
        self.vfh              = VFH(**vfh_kwargs)
        self.base_speed       = BASE_SPEED
        self.min_speed        = MIN_SPEED
        self.rotation_trigger = ROTATION_TRIGGER

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(
        self,
        depth_map: np.ndarray,
        goal_bearing_deg: float = 0.0,
        check_sonar: bool = True,
    ) -> VFHResult | None:
        """
        Apply one frame of VFH navigation.

        Parameters
        ----------
        depth_map        : (H, W) float32 from InferencePipeline, 0=far 1=near
        goal_bearing_deg : desired heading in degrees (0 = forward, + = right)
        check_sonar      : if False, skip the sonar STOP/SLOW check — used by the
                           state machine in APPROACHING so it can manage its own
                           stop distance rather than halting at the STOP zone

        Returns
        -------
        VFHResult  — the chosen steering direction and histogram (for logging/debug)
        None       — all sectors blocked; the robot is now rotating left to rescan
        """
        # ── Sonar safety check (hard-stop / slow-down) ──────────────────────
        if check_sonar and self.sonar is not None:
            sonar_zone = self.sonar.zone
            if sonar_zone == ZONE_STOP:
                self.bot.stop()
                log.debug("Sonar STOP (%.1f cm) — halting",
                          self.sonar.distance_cm or 0)
                return None
        else:
            sonar_zone = None if not check_sonar else (
                self.sonar.zone if self.sonar else None
            )

        result = self.vfh.compute(depth_map, goal_bearing_deg)

        if result is None:
            log.debug("VFH: all sectors blocked — rotating to rescan")
            self.bot.rotate("left", ROTATION_SPEED)
            return None

        if abs(result.angle_deg) > self.rotation_trigger:
            # Obstacle is off to one side: rotate the chassis to face the clear
            # direction so the camera realigns with the chosen path.  Translating
            # without rotating on an omni-drive just keeps the obstacle in frame.
            rot_dir = "right" if result.angle_deg > 0 else "left"
            self.bot.rotate(rot_dir, ROTATION_SPEED)
            log.debug("VFH: angle=%+.1f° > %.0f° threshold — rotating %s",
                      result.angle_deg, self.rotation_trigger, rot_dir)
        else:
            speed = self._speed_from_clearance(result)
            if sonar_zone == ZONE_SLOW:
                speed = min(speed, SLOW_ZONE_CAP)
                log.debug("Sonar SLOW (%.1f cm) — capping speed to %d%%",
                          self.sonar.distance_cm or 0, speed)
            self.bot.startMove([result.vx, result.vy], speed)
            log.debug(
                "VFH: angle=%+.1f°  vx=%+.2f  vy=%.2f  speed=%d%%",
                result.angle_deg, result.vx, result.vy, speed,
            )
        return result

    # ── Standalone test loop ──────────────────────────────────────────────────

    def run(
        self,
        pipeline,                       # InferencePipeline instance
        goal_bearing_deg: float = 0.0,
        duration_s: float | None = None,
        verbose: bool = True,
        target_label: str | None = None,
        camera_fov_deg: float = 62.0,
    ) -> None:
        """
        Drive the robot using live depth frames until duration_s elapses
        or KeyboardInterrupt.

        Parameters
        ----------
        pipeline         : started InferencePipeline
        goal_bearing_deg : fallback heading bias when no detection is visible (degrees)
        duration_s       : run time in seconds; None = run forever
        verbose          : print status each frame
        target_label     : beverage label to seek; None = any detection
        camera_fov_deg   : camera horizontal FOV for bearing computation
        """
        from inference_pipeline import goal_bearing_from_detections

        interval  = 1.0 / LOOP_HZ
        deadline  = None if duration_s is None else time.monotonic() + duration_s
        frame_n   = 0

        log.info(
            "Navigator running — goal=%.1f°  target=%s  duration=%s",
            goal_bearing_deg,
            target_label or "any",
            f"{duration_s:.0f}s" if duration_s else "∞",
        )

        try:
            while True:
                if deadline and time.monotonic() >= deadline:
                    log.info("Duration elapsed — stopping.")
                    break

                t0 = time.monotonic()
                _, depth, detections = pipeline.get_state()

                if depth is None:
                    time.sleep(0.05)
                    continue

                # Use detected object bearing when visible, else fall back to
                # the fixed goal_bearing_deg (default 0 = straight ahead).
                bearing = goal_bearing_from_detections(
                    detections, target_label, camera_fov_deg
                )
                active_goal = bearing if bearing is not None else goal_bearing_deg

                result = self.step(depth, active_goal)
                frame_n += 1

                if verbose:
                    sonar_str = ""
                    if self.sonar is not None:
                        d = self.sonar.distance_cm
                        z = self.sonar.zone
                        sonar_str = f"  sonar={d:.0f}cm [{z}]" if d is not None else "  sonar=---"
                    target_str = f"  goal={active_goal:+.1f}°"
                    target_str += " [locked]" if bearing is not None else " [no target]"
                    if result is None:
                        print(f"\rFrame {frame_n:4d} | STOP/BLOCKED — rotating{target_str}{sonar_str}   ",
                              end="", flush=True)
                    elif abs(result.angle_deg) > self.rotation_trigger:
                        print(
                            f"\rFrame {frame_n:4d} | "
                            f"angle={result.angle_deg:+6.1f}°  "
                            f"ROTATING {'right' if result.angle_deg > 0 else 'left '}"
                            f"{target_str}{sonar_str}   ",
                            end="", flush=True,
                        )
                    else:
                        speed = self._speed_from_clearance(result)
                        if self.sonar is not None and self.sonar.zone == ZONE_SLOW:
                            speed = min(speed, SLOW_ZONE_CAP)
                        print(
                            f"\rFrame {frame_n:4d} | "
                            f"angle={result.angle_deg:+6.1f}°  "
                            f"vx={result.vx:+.2f}  vy={result.vy:.2f}  "
                            f"speed={speed:3d}%{target_str}{sonar_str}   ",
                            end="", flush=True,
                        )

                # Pace the loop to LOOP_HZ
                elapsed = time.monotonic() - t0
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.bot.stop()
            print()  # newline after the \r status line
            log.info("Navigator stopped — bot halted.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _speed_from_clearance(self, result: VFHResult) -> int:
        """
        Scale speed by the fraction of sectors that are passable.

        All sectors open (fraction=1.0) → BASE_SPEED
        Heavily cluttered  (fraction→0) → MIN_SPEED

        Using the passable fraction rather than mean clearance value avoids
        the per-frame normalisation problem: in a clear scene the ground plane
        always produces the highest depth value (nearest=1.0), which would
        depress the absolute clearance reading even with no real obstacles.
        The fraction of passable sectors is a more reliable "how open is the
        scene" signal.
        """
        passable_fraction = float(result.valley_mask.mean())
        speed = self.min_speed + (self.base_speed - self.min_speed) * passable_fraction
        return max(self.min_speed, min(self.base_speed, int(speed)))


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from inference_pipeline import InferencePipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="VFH obstacle-avoidance test drive")
    parser.add_argument("--duration",  type=float, default=30.0,
                        help="seconds to run (default: 30)")
    parser.add_argument("--goal",      type=float, default=0.0,
                        help="goal bearing in degrees (default: 0 = straight)")
    parser.add_argument("--base-speed",  type=int,   default=BASE_SPEED,
                        help=f"max drive speed %% (default: {BASE_SPEED})")
    parser.add_argument("--min-speed",   type=int,   default=MIN_SPEED,
                        help=f"min drive speed %% (default: {MIN_SPEED})")
    parser.add_argument("--rot-trigger",  type=float, default=ROTATION_TRIGGER,
                        help=f"steer angle (°) above which robot rotates in place (default: {ROTATION_TRIGGER})")
    parser.add_argument("--ground-crop",  type=float, default=0.30,
                        help="fraction of image height to drop as ground plane (default: 0.30)")
    parser.add_argument("--threshold",    type=float, default=0.65,
                        help="VFH safe_threshold — lower = react sooner (default: 0.65)")
    parser.add_argument("--sonar",        action="store_true",
                        help="enable HC-SR04 safety guard (GPIO 5/6)")
    parser.add_argument("--target",       type=str, default=None,
                        help="beverage label to seek (e.g. 'bottle-plastic'); "
                             "default: steer toward any detection")
    parser.add_argument("--fov",          type=float, default=62.0,
                        help="camera horizontal FOV in degrees (default: 62.0)")
    args = parser.parse_args()

    pipeline = InferencePipeline()
    pipeline.start()

    from inference_pipeline import goal_bearing_from_detections

    # Wait for the first depth frame before moving
    log.info("Waiting for first depth frame …")
    while True:
        _, depth, _ = pipeline.get_state()
        if depth is not None:
            break
        time.sleep(0.1)
    log.info("First depth frame received — starting navigation.")

    sonar = None
    if args.sonar:
        sonar = SonarGuard()
        sonar.start()

    bot = OmniBot()
    nav = Navigator(bot,
                    sonar=sonar,
                    base_speed=args.base_speed,
                    min_speed=args.min_speed,
                    ground_crop=args.ground_crop,
                    safe_threshold=args.threshold)
    nav.rotation_trigger = args.rot_trigger

    try:
        nav.run(pipeline,
                goal_bearing_deg=args.goal,
                duration_s=args.duration,
                target_label=args.target,
                camera_fov_deg=args.fov)
    finally:
        if sonar:
            sonar.stop()
        pipeline.stop()
