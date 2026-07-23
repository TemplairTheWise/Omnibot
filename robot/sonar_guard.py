"""
sonar_guard.py — Step 4: HC-SR04 background safety thread.

Reads the ultrasonic sensor in a daemon thread and classifies the
current distance into three zones:

    CLEAR  (>= slow_cm)          — full speed allowed
    SLOW   (stop_cm .. slow_cm)  — cap forward speed
    STOP   (< stop_cm)           — do not move forward

The Navigator queries zone / distance_cm each step; this module never
issues bot commands directly so Navigator stays the sole command authority.

Public API
----------
    guard = SonarGuard()
    guard.start()

    zone = guard.zone           # ZONE_CLEAR | ZONE_SLOW | ZONE_STOP
    dist = guard.distance_cm    # latest reading in cm, or None before first read

    guard.stop()                # shut down cleanly (closes GPIO)

Standalone test
---------------
    python sonar_guard.py
"""

import logging
import statistics
import threading
import time
from collections import deque

log = logging.getLogger(__name__)

# ── GPIO pins (BCM numbering, same as distance.py) ────────────────────────────
TRIGGER_PIN = 5
ECHO_PIN    = 6

# ── Distance thresholds ───────────────────────────────────────────────────────
STOP_CM       = 20.0  # hard-stop zone
SLOW_CM       = 60.0  # slow-down zone
POLL_HZ       = 10    # sensor reads per second
MIN_VALID_CM  = 4.0   # readings below this are hardware noise (HC-SR04 min range ~2 cm;
                       # soft-PWM timing errors often produce spurious 0–4 cm readings)
MEDIAN_WINDOW = 5     # number of valid readings to median-filter
STOP_COUNT    = 3     # consecutive filtered readings below stop_cm required to enter STOP
                       # (at 10 Hz → 300 ms;  prevents a single dip from halting the robot)

# ── Zone constants ────────────────────────────────────────────────────────────
ZONE_CLEAR = "clear"
ZONE_SLOW  = "slow"
ZONE_STOP  = "stop"


class SonarGuard:
    """
    Runs a daemon thread that polls the HC-SR04 and maintains a thread-safe
    distance reading and zone classification.
    """

    def __init__(
        self,
        trigger_pin:   int   = TRIGGER_PIN,
        echo_pin:      int   = ECHO_PIN,
        stop_cm:       float = STOP_CM,
        slow_cm:       float = SLOW_CM,
        poll_hz:       float = POLL_HZ,
        min_valid_cm:  float = MIN_VALID_CM,
        median_window: int   = MEDIAN_WINDOW,
        stop_count:    int   = STOP_COUNT,
    ):
        # Import here so the module can be imported on machines without GPIO
        # hardware (unit tests, CI) without an immediate crash.
        from gpiozero import DistanceSensor
        self._sensor        = DistanceSensor(echo=echo_pin, trigger=trigger_pin,
                                             max_distance=4.0)
        self._stop_cm       = stop_cm
        self._slow_cm       = slow_cm
        self._interval      = 1.0 / poll_hz
        self._min_valid_cm  = min_valid_cm
        self._buf: deque[float] = deque(maxlen=median_window)
        self._stop_count    = stop_count
        self._stop_streak   = 0   # consecutive filtered readings in STOP zone
        self._lock          = threading.Lock()
        self._dist_cm: float | None = None
        self._zone          = ZONE_CLEAR
        self._running       = False
        self._thread:  threading.Thread | None = None

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def distance_cm(self) -> float | None:
        """Latest distance reading in centimetres, or None before first read."""
        with self._lock:
            return self._dist_cm

    @property
    def zone(self) -> str:
        """Current zone: ZONE_CLEAR, ZONE_SLOW, or ZONE_STOP."""
        with self._lock:
            return self._zone

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="SonarGuard", daemon=True
        )
        self._thread.start()
        log.info(
            "SonarGuard started — stop=%.0f cm, slow=%.0f cm, "
            "min_valid=%.0f cm, window=%d, stop_count=%d",
            self._stop_cm, self._slow_cm,
            self._min_valid_cm, self._buf.maxlen, self._stop_count,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sensor.close()
        log.info("SonarGuard stopped.")

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                raw_cm = self._sensor.distance * 100.0
            except Exception as exc:
                log.warning("Sonar read error: %s", exc)
                time.sleep(self._interval)
                continue

            # Drop readings below the sensor's minimum reliable range.
            # Soft-PWM timing jitter often produces spurious 0–4 cm bursts
            # that would otherwise trigger a false STOP.
            if raw_cm < self._min_valid_cm:
                log.debug("Sonar: %.1f cm — below min_valid (%.0f cm), skipped",
                          raw_cm, self._min_valid_cm)
                elapsed = time.monotonic() - t0
                if self._interval - elapsed > 0:
                    time.sleep(self._interval - elapsed)
                continue

            # Median filter over the last N valid readings.
            self._buf.append(raw_cm)
            dist_cm = statistics.median(self._buf)

            # Classify zone; STOP requires consecutive filtered readings to
            # avoid a single echo-bounce dip halting the robot.
            if dist_cm < self._stop_cm:
                self._stop_streak += 1
                zone = ZONE_STOP if self._stop_streak >= self._stop_count else ZONE_SLOW
            else:
                self._stop_streak = 0
                zone = ZONE_SLOW if dist_cm < self._slow_cm else ZONE_CLEAR

            with self._lock:
                self._dist_cm = dist_cm
                self._zone    = zone

            if zone != ZONE_CLEAR:
                log.debug("Sonar: raw=%.1f cm  filtered=%.1f cm — %s",
                          raw_cm, dist_cm, zone)

            elapsed = time.monotonic() - t0
            sleep_t = self._interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    guard = SonarGuard()
    guard.start()
    print("HC-SR04 live readings — Ctrl+C to stop\n")
    try:
        while True:
            d = guard.distance_cm
            z = guard.zone
            bar = {"clear": ".", "slow": "~", "stop": "X"}[z]
            dist_str = f"{d:5.1f} cm" if d is not None else "  --- "
            print(f"\r{bar} {dist_str}  [{z:5s}]", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        guard.stop()
