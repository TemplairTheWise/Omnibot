"""
state_machine.py — Step 7: Autonomous navigation state machine.

Ties PolarScan, Navigator (VFH + sonar), and detection together into a
single background thread the Flask API can start, stop, and query.

States
------
  IDLE        — waiting for start()
  SCANNING    — executing 360° polar scan
  SEARCHING   — VFH obstacle avoidance, watching for the target
  APPROACHING — target locked; steering toward it; sonar caps speed
  FOUND       — robot stopped at the target

Transitions
-----------
  IDLE        --start()-->             SCANNING (or SEARCHING if skip_scan)
  SCANNING    --target seen-->         APPROACHING
  SCANNING    --scan complete-->       SEARCHING
  SEARCHING   --target detected-->     APPROACHING
  SEARCHING   --no detection N secs--> SCANNING  (periodic re-scan)
  APPROACHING --arrived at target-->   FOUND
  APPROACHING --target lost-->         SEARCHING
  FOUND / any --stop()-->              IDLE

"Arrived" is defined as:
  • sonar in STOP zone (<20 cm) AND target still visible, OR
  • target bounding box covers ≥ FOUND_BOX_AREA of the frame

Public API
----------
  nsm = NavStateMachine(bot, pipeline, sonar)
  nsm.start(target_label="bottle-plastic")
  status = nsm.get_status()      # JSON-safe dict for Flask
  history = nsm.get_history()    # list of past session summaries
  nsm.stop()

Standalone test (hardware required)
------------------------------------
  python state_machine.py [--target bottle-plastic] [--no-scan]
"""

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_pipeline import goal_bearing_from_detections
from navigator import LOOP_HZ, ROTATION_SPEED, Navigator
from omnibot import OmniBot
from polar_scan import PolarScan
from sonar_guard import ZONE_STOP, SonarGuard

log = logging.getLogger(__name__)

# ── State labels ──────────────────────────────────────────────────────────────
IDLE        = "IDLE"
SCANNING    = "SCANNING"
SEARCHING   = "SEARCHING"
APPROACHING = "APPROACHING"
FOUND       = "FOUND"

# ── Tunables ──────────────────────────────────────────────────────────────────
TARGET_LOST_FRAMES   = 50    # consecutive no-detection frames → "target lost"
SEARCH_LOOP_HZ       = 10   # state-machine update rate

# APPROACHING-specific
APPROACH_STOP_CM        = 10.0
APPROACH_SLOW_CM        = 25.0
APPROACH_BASE_SPEED     = 55
APPROACH_SLOW_SPEED     = 35
APPROACH_ROTATE_BEARING = 8.0
FOUND_BOX_AREA          = 0.25

# Re-scan: if no detection for this many seconds while SEARCHING, stop and
# run a new polar scan.  Prevents the robot from wandering indefinitely.
RESCAN_AFTER_S = 20.0

# Session logs directory
LOG_DIR = Path("search_logs")


class NavStateMachine:
    """
    Runs the full navigation behaviour in a daemon thread.
    Thread-safe: start() / stop() / get_status() / get_history() can be
    called from any thread.
    """

    def __init__(
        self,
        bot:       OmniBot,
        pipeline,
        sonar:     SonarGuard | None = None,
        skip_scan: bool = False,
        camera_fov_deg: float = 62.0,
        **kwargs,
    ):
        self.bot            = bot
        self.pipeline       = pipeline
        self.sonar          = sonar
        self.skip_scan      = skip_scan
        self.camera_fov_deg = camera_fov_deg

        nav_keys  = {k: v for k, v in kwargs.items()
                     if k in ("base_speed", "min_speed", "rotation_trigger",
                               "ground_crop", "safe_threshold", "n_sectors",
                               "smoothing_sigma", "goal_bias")}
        scan_keys = {k: v for k, v in kwargs.items()
                     if k in ("n_steps", "rotation_speed", "deg_per_sec",
                               "settle_s", "stop_on_target")}

        self.navigator = Navigator(bot, sonar=sonar, **nav_keys)
        self.scanner   = PolarScan(bot, pipeline, **scan_keys)

        self._lock          = threading.Lock()
        self._state         = IDLE
        self._target_label: str | None = None
        self._last_det_label: str | None = None
        self._last_det_conf: float = 0.0
        self._last_bearing: float | None = None
        self._stop_event    = threading.Event()
        self._thread: threading.Thread | None = None

        # Session logging
        self._session: dict | None = None
        self._session_start_t: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, target_label: str | None = None,
              record_path: str | None = None,
              record_flip: bool = False,
              skip_scan: bool | None = None) -> bool:
        with self._lock:
            if self._state != IDLE:
                log.warning("start() called while in state %s — ignored", self._state)
                return False
            self._target_label   = target_label
            self._last_det_label = None
            self._last_det_conf  = 0.0
            self._last_bearing   = None
            self._stop_event.clear()

            # Initialise session log
            self._session_start_t = time.monotonic()
            self._session = {
                "session_id":       datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                "target":           target_label or "any",
                "start_time":       datetime.now(timezone.utc).isoformat(),
                "end_time":         None,
                "duration_s":       None,
                "outcome":          None,
                "skip_scan":        bool(self.skip_scan if skip_scan is None else skip_scan),
                "events":           [],
                "detections_count": 0,
            }

            _skip = self.skip_scan if skip_scan is None else skip_scan
            self._set_state(SCANNING if not _skip else SEARCHING)

        if record_path:
            self.pipeline.start_recording(record_path, flip=record_flip)

        self._thread = threading.Thread(
            target=self._run, name="NavStateMachine", daemon=True
        )
        self._thread.start()
        log.info("NavStateMachine started — target=%s  skip_scan=%s  record=%s",
                 target_label or "any", skip_scan, record_path or "off")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self.bot.stop()
        self.pipeline.stop_recording()
        with self._lock:
            self._set_state(IDLE)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("NavStateMachine stopped.")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            sonar_cm = None
            if self.sonar is not None:
                sonar_cm = self.sonar.distance_cm
            return {
                "state":      self._state,
                "target":     self._target_label,
                "detected":   self._last_det_label,
                "confidence": round(self._last_det_conf, 2),
                "bearing_deg": (round(self._last_bearing, 1)
                                if self._last_bearing is not None else None),
                "sonar_cm":   (round(sonar_cm, 1)
                               if sonar_cm is not None else None),
            }

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return summaries of past sessions, newest first."""
        if not LOG_DIR.exists():
            return []
        files = sorted(LOG_DIR.glob("*.json"), reverse=True)[:limit]
        result = []
        for f in files:
            try:
                data = json.loads(f.read_text())
                result.append({
                    "session_id":       data.get("session_id"),
                    "target":           data.get("target"),
                    "start_time":       data.get("start_time"),
                    "duration_s":       data.get("duration_s"),
                    "outcome":          data.get("outcome"),
                    "detections_count": data.get("detections_count", 0),
                })
            except Exception:
                pass
        return result

    # ── Session logging helpers ───────────────────────────────────────────────

    def _log_event(self, kind: str, **kw) -> None:
        """Append an event to the current session.  Lock-free; call from run thread only."""
        if self._session is None:
            return
        entry = {"t": round(time.monotonic() - self._session_start_t, 2), "type": kind}
        entry.update(kw)
        self._session["events"].append(entry)

    def _finalize_session(self, outcome: str) -> None:
        """Write the session JSON to disk.  Called from the run thread's finally block."""
        if self._session is None:
            return
        self._session["end_time"]   = datetime.now(timezone.utc).isoformat()
        self._session["duration_s"] = round(time.monotonic() - self._session_start_t, 1)
        self._session["outcome"]    = outcome
        self._session["detections_count"] = sum(
            1 for e in self._session["events"] if e["type"] == "detection"
        )
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{self._session['session_id']}.json"
        try:
            path.write_text(json.dumps(self._session, indent=2))
            log.info("Session log → %s  outcome=%s  dur=%.1fs  det=%d",
                     path.name, outcome,
                     self._session["duration_s"],
                     self._session["detections_count"])
        except OSError as exc:
            log.warning("Could not write session log: %s", exc)
        self._session = None

    # ── Internal: main thread ─────────────────────────────────────────────────

    def _run(self) -> None:
        found = False
        try:
            if self._current_state() == SCANNING:
                self._scanning_phase()
            if self._stop_event.is_set():
                return
            if self._current_state() in (SEARCHING, APPROACHING):
                found = self._searching_phase()
        except Exception:
            log.exception("NavStateMachine: unhandled error in run loop")
        finally:
            self.bot.stop()
            with self._lock:
                if self._state not in (IDLE, FOUND):
                    self._set_state(IDLE)

            if found:
                outcome = "FOUND"
            elif self._stop_event.is_set():
                outcome = "STOPPED"
            else:
                outcome = "TIMEOUT"
            self._finalize_session(outcome)
            log.info("NavStateMachine run thread exiting — state=%s  outcome=%s",
                     self._current_state(), outcome)

    # ── Phase: SCANNING ───────────────────────────────────────────────────────

    def _scanning_phase(self) -> None:
        log.info("Phase: SCANNING  target=%s", self._target_label)
        result = self.scanner.run(target_label=self._target_label)

        if self._stop_event.is_set():
            return

        if result.target_heading_deg is not None:
            log.info("Scan: target found at %.1f° — APPROACHING",
                     result.target_heading_deg)
            with self._lock:
                self._set_state(APPROACHING)
        else:
            log.info("Scan: no target — facing %.1f°, going to SEARCHING",
                     result.best_heading_deg)
            self.scanner.face_heading(result.best_heading_deg)
            with self._lock:
                self._set_state(SEARCHING)

    # ── Inline re-scan (called from within _searching_phase) ─────────────────

    def _inline_rescan(self) -> bool:
        """
        Run a polar scan in-place and update state.
        Returns True if the target was found (state → APPROACHING).
        Called from the searching loop; run thread only.
        """
        log.info("Re-scanning after %.0fs without detection", RESCAN_AFTER_S)
        self.bot.stop()
        time.sleep(0.3)
        if self._stop_event.is_set():
            return False

        with self._lock:
            self._set_state(SCANNING)

        result = self.scanner.run(target_label=self._target_label)
        if self._stop_event.is_set():
            return False

        if result.target_heading_deg is not None:
            log.info("Re-scan: target at %.1f° — APPROACHING",
                     result.target_heading_deg)
            with self._lock:
                self._set_state(APPROACHING)
            return True
        else:
            log.info("Re-scan: no target — facing %.1f°", result.best_heading_deg)
            self.scanner.face_heading(result.best_heading_deg)
            with self._lock:
                self._set_state(SEARCHING)
            return False

    # ── Phase: SEARCHING / APPROACHING (shared loop) ──────────────────────────

    def _searching_phase(self) -> bool:
        """
        Unified loop handling SEARCHING and APPROACHING.

        Returns True if the robot reached FOUND, False otherwise.
        """
        interval              = 1.0 / SEARCH_LOOP_HZ
        lost_frames           = 0
        last_approach_bearing: float | None = None
        last_search_det_t     = time.monotonic()  # time of last detection in SEARCHING

        while not self._stop_event.is_set():
            t0    = time.monotonic()
            state = self._current_state()

            _, depth, detections = self.pipeline.get_state()
            if depth is None:
                time.sleep(0.05)
                continue

            # ── Target bearing ────────────────────────────────────────────────
            bearing = goal_bearing_from_detections(
                detections, self._target_label, self.camera_fov_deg
            )

            with self._lock:
                if bearing is not None:
                    best = max(
                        [d for d in detections
                         if self._target_label is None or d.label == self._target_label],
                        key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1),
                    )
                    self._last_det_label = best.label
                    self._last_det_conf  = best.confidence
                    self._last_bearing   = bearing
                    # Log detection event (lock-free inside the lock block is fine
                    # because _log_event only touches _session, not the lock)
                    self._log_event("detection",
                                    label=best.label,
                                    confidence=round(best.confidence, 2),
                                    bearing=round(bearing, 1),
                                    state=state)
                else:
                    self._last_bearing = None

            # ── State transitions ─────────────────────────────────────────────
            just_entered_approaching = False

            if state == SEARCHING:
                last_approach_bearing = None

                if bearing is not None:
                    log.info("Target detected at %.1f° — APPROACHING", bearing)
                    with self._lock:
                        self._set_state(APPROACHING)
                    last_approach_bearing = bearing
                    lost_frames = 0
                    last_search_det_t = time.monotonic()
                    just_entered_approaching = True
                else:
                    # Periodic re-scan when target hasn't been seen for a while
                    if time.monotonic() - last_search_det_t > RESCAN_AFTER_S:
                        target_found = self._inline_rescan()
                        last_search_det_t = time.monotonic()
                        if target_found:
                            last_approach_bearing = None  # no live bearing yet
                            just_entered_approaching = True
                            lost_frames = 0

            elif state == APPROACHING:
                if bearing is not None:
                    last_approach_bearing = bearing
                    lost_frames = 0

                    # ── Arrival check ─────────────────────────────────────────
                    arrived = False
                    if self.sonar is not None:
                        dist = self.sonar.distance_cm
                        if dist is not None and dist <= APPROACH_STOP_CM:
                            arrived = True

                    if not arrived:
                        candidates = [
                            d for d in detections
                            if self._target_label is None
                            or d.label == self._target_label
                        ]
                        if candidates:
                            best = max(candidates,
                                       key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
                            if (best.x2 - best.x1) * (best.y2 - best.y1) >= FOUND_BOX_AREA:
                                arrived = True

                    if arrived:
                        self.bot.stop()
                        with self._lock:
                            self._set_state(FOUND)
                        log.info("FOUND — target reached!")
                        return True

                else:
                    lost_frames += 1
                    if lost_frames >= TARGET_LOST_FRAMES:
                        log.info("Target lost for %d frames — back to SEARCHING",
                                 lost_frames)
                        with self._lock:
                            self._set_state(SEARCHING)
                        lost_frames = 0
                        last_approach_bearing = None
                        last_search_det_t = time.monotonic()  # reset re-scan timer

            # ── Drive command ─────────────────────────────────────────────────
            if state == APPROACHING or just_entered_approaching:
                steer = bearing if bearing is not None else last_approach_bearing
                speed = APPROACH_BASE_SPEED
                if self.sonar is not None:
                    dist = self.sonar.distance_cm
                    if dist is not None and dist <= APPROACH_SLOW_CM:
                        speed = APPROACH_SLOW_SPEED

                if bearing is not None and steer is not None:
                    # Negate: camera is 180° rotated
                    corrected = -steer
                    if abs(corrected) > APPROACH_ROTATE_BEARING:
                        rot_dir = "right" if corrected > 0 else "left"
                        self.bot.rotate(rot_dir, ROTATION_SPEED)
                    else:
                        self.bot.startMove([0.0, 1.0], speed)
                else:
                    self.bot.startMove([0.0, 1.0], speed)
            else:
                active_goal = bearing if bearing is not None else 0.0
                self.navigator.step(depth, active_goal)

            # ── Pace the loop ─────────────────────────────────────────────────
            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        return False  # stop_event was set

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        """Set state; must be called with _lock held."""
        if self._state != new_state:
            log.info("State: %s → %s", self._state, new_state)
            self._state = new_state
            self._log_event("state", state=new_state)

    def _current_state(self) -> str:
        with self._lock:
            return self._state


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from inference_pipeline import InferencePipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Navigation state machine test")
    parser.add_argument("--target",   type=str,  default=None)
    parser.add_argument("--no-scan",  action="store_true")
    parser.add_argument("--sonar",    action="store_true")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--record",   type=str,  default=None, metavar="FILE.mp4")
    parser.add_argument("--flip",     action="store_true")
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

    sonar = None
    if args.sonar:
        from sonar_guard import SonarGuard
        sonar = SonarGuard()
        sonar.start()

    bot = OmniBot()
    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=args.no_scan)
    nsm.start(target_label=args.target, record_path=args.record,
              record_flip=args.flip)

    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            status = nsm.get_status()
            det_str   = (f"{status['detected']} {status['confidence']:.0%}"
                         if status["detected"] else "—")
            sonar_str = (f"  sonar={status['sonar_cm']:.0f}cm"
                         if status["sonar_cm"] is not None else "")
            print(
                f"\r[{status['state']:11s}]  target={status['target'] or 'any':<16s}"
                f"  det={det_str:<25s}{sonar_str}   ",
                end="", flush=True,
            )
            if status["state"] in (FOUND, IDLE):
                print()
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        nsm.stop()
        if sonar:
            sonar.stop()
        pipeline.stop()
