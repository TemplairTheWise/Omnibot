"""
eval_navigation.py — Step 14: Interactive navigation trial runner.

Run this script with the robot physically present.  For each trial it:
  1. Prompts you for the scenario name and expected target.
  2. Optionally records the run - the video is saved automatically under
     attempts/ with a generated filename, no path to type.
  3. Starts the NavStateMachine (with or without polar scan).
  4. Waits until the robot reaches FOUND or you abort with Ctrl-C.
  5. Prompts you to confirm success and note any observations.
  6. Writes one row to eval_navigation_results.csv.

After all trials it prints a summary table.

Usage
-----
    python eval_navigation.py [--trials N] [--scan] [--timeout 120]

Output
------
  eval_navigation_results.csv  (appended, so runs accumulate across sessions)
  attempts/*.mp4               (only for trials you choose to record)

Requirements
-----------
  • Hailo hardware + motors connected
  • Source setup_env.sh first
"""

import argparse
import csv
import logging
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

CSV_PATH     = Path("eval_navigation_results.csv")
ATTEMPTS_DIR = Path("attempts")
FIELDNAMES = [
    "trial", "scenario", "target", "success", "outcome",
    "duration_s", "scan_s", "detections_count", "session_id", "recording", "notes",
]
# duration_s excludes time spent in the polar scan (initial + any re-scans) -
# see scan_s for how much was excluded. Pulled from the session log's
# nav_duration_s, not the raw polled elapsed time, so it isn't inflated by
# scan cost that's unrelated to the actual search/approach being measured.


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "trial"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        return

    # If FIELDNAMES changed since this file was created, appending now would
    # silently misalign every column from the change onward (happened once
    # already when scan_s was added) - refuse instead of writing bad rows.
    with CSV_PATH.open(newline="") as f:
        existing_header = next(csv.reader(f), [])
    if existing_header != FIELDNAMES:
        raise SystemExit(
            f"{CSV_PATH} has an outdated header - the column layout changed "
            f"since this file was created, so appending would silently "
            f"misalign every row.\n"
            f"  File header : {existing_header}\n"
            f"  Expected    : {FIELDNAMES}\n"
            f"Migrate or rename the existing file before running more trials."
        )


def _append_row(row: dict) -> None:
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def _count_existing_trials() -> int:
    if not CSV_PATH.exists():
        return 0
    with CSV_PATH.open() as f:
        return max(0, sum(1 for _ in f) - 1)   # minus header


# ── Status polling ────────────────────────────────────────────────────────────

def _wait_for_terminal(nsm, timeout_s: float) -> tuple[str, float]:
    """
    Poll NSM until it reaches FOUND or IDLE (stopped), or timeout elapses.
    Returns (final_state, elapsed_seconds).
    """
    terminal = {"FOUND", "IDLE"}
    t0 = time.monotonic()
    while True:
        status = nsm.get_status()
        state  = status["state"]

        elapsed = time.monotonic() - t0
        det_str = (f"{status['detected']} {status['confidence']:.0%}"
                   if status.get("detected") else "—")
        sonar   = (f"  sonar={status['sonar_cm']:.0f}cm"
                   if status.get("sonar_cm") is not None else "")
        print(
            f"\r  [{state:11s}]  det={det_str:<25s}{sonar}  {elapsed:.0f}s  ",
            end="", flush=True,
        )

        if state in terminal or elapsed >= timeout_s:
            print()
            return state, elapsed

        time.sleep(0.5)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(nsm, trial_no: int, do_scan: bool, timeout_s: float) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Trial {trial_no}")
    print(f"{'─'*60}")

    scenario = input("  Scenario description (e.g. 'open table, 1 bottle'): ").strip()
    target   = input("  Target label (blank = any beverage): ").strip() or None

    record = None
    if input("  Record this trial? [y/N]: ").strip().lower() in ("y", "yes"):
        ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%H%M%S")
        fname = f"trial{trial_no:03d}_{_slugify(scenario)}_{ts}.mp4"
        record = str(ATTEMPTS_DIR / fname)
        print(f"  Recording → {record}")

    print(f"\n  Starting — scan={'yes' if do_scan else 'no'}  timeout={timeout_s}s")
    print("  Press Ctrl-C to abort the trial early.\n")

    t_start = time.monotonic()
    ok = nsm.start(
        target_label  = target,
        record_path   = record,
        record_flip   = True,
        skip_scan     = not do_scan,
    )
    if not ok:
        print("  [!] NSM failed to start — robot may be mid-state.  Skipping trial.")
        return {}

    aborted = False
    try:
        final_state, elapsed = _wait_for_terminal(nsm, timeout_s)
    except KeyboardInterrupt:
        print("\n  [Ctrl-C] Aborting trial …")
        nsm.stop()
        final_state, elapsed = "STOPPED", time.monotonic() - t_start
        aborted = True

    if not aborted:
        # stop() joins the run thread, which guarantees the session JSON has
        # been flushed to search_logs/ before get_history() below reads it -
        # without this, get_history() can race the background thread and
        # return the *previous* trial's session instead of this one's.
        nsm.stop()

    # Grab detections count + session id + scan-excluded duration from the
    # last session log
    history        = nsm.get_history(limit=1)
    det_count      = history[0]["detections_count"] if history else 0
    session_id     = history[0]["session_id"] if history else None
    scan_s         = history[0].get("scan_s") if history else None
    nav_duration_s = history[0].get("nav_duration_s") if history else None

    # Prefer the session log's navigation-only duration (excludes polar-scan
    # time); fall back to the raw polled elapsed time if the log is missing.
    duration_reported = nav_duration_s if nav_duration_s is not None else round(elapsed, 1)

    if scan_s is not None:
        print(f"\n  Final state : {final_state}  "
              f"(total {elapsed:.1f}s, nav {duration_reported:.1f}s, scan {scan_s:.1f}s)")
    else:
        print(f"\n  Final state : {final_state}  ({elapsed:.1f} s)")
    print(f"  Detections  : {det_count}")

    success_raw = input("  Mark as SUCCESS? [y/N]: ").strip().lower()
    success     = success_raw in ("y", "yes")
    notes       = input("  Notes (optional): ").strip()

    return {
        "trial":            trial_no,
        "scenario":         scenario,
        "target":           target or "any",
        "success":          "1" if success else "0",
        "outcome":          final_state,
        "duration_s":       duration_reported,
        "scan_s":           scan_s if scan_s is not None else "",
        "detections_count": det_count,
        "session_id":       session_id or "",
        "recording":        record or "",
        "notes":            notes,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    n_ok    = sum(1 for r in rows if r.get("success") == "1")
    n_total = len(rows)
    dur_avg = sum(float(r["duration_s"]) for r in rows) / n_total

    print(f"\n{'═'*60}")
    print(f"  Session summary  —  {n_ok}/{n_total} trials successful")
    print(f"  Avg duration     : {dur_avg:.1f} s")
    print(f"{'─'*60}")
    fmt = "  {:>5}  {:<12}  {:<20}  {:>7}  {:>4}"
    print(fmt.format("#", "Outcome", "Scenario", "Dur(s)", "OK?"))
    print(f"{'─'*60}")
    for r in rows:
        print(fmt.format(
            r["trial"],
            r["outcome"][:12],
            r["scenario"][:20],
            r["duration_s"],
            "✓" if r["success"] == "1" else "✗",
        ))
    print(f"{'═'*60}\n")
    print(f"  Results appended to {CSV_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Interactive navigation trial runner"
    )
    parser.add_argument("--trials",  type=int,   default=5,
                        help="Number of trials to run (default 5)")
    parser.add_argument("--scan",    action="store_true",
                        help="Enable polar scan before each search")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-trial navigation timeout in seconds - if "
                             "--scan is set, the initial polar scan's "
                             "estimated duration is added on top of this "
                             "automatically (default 120)")
    args = parser.parse_args()

    # Hardware init
    print("\nInitialising hardware …")
    from polar_scan import N_STEPS, DEG_PER_SEC, SETTLE_S

    effective_timeout = args.timeout
    if args.scan:
        step_angle_deg = 360.0 / N_STEPS
        scan_s = N_STEPS * (step_angle_deg / DEG_PER_SEC + SETTLE_S)
        effective_timeout += scan_s
        print(f"--scan enabled — adding ~{scan_s:.0f}s estimated scan time "
              f"to the {args.timeout:.0f}s timeout → {effective_timeout:.0f}s per trial")
    from omnibot import OmniBot
    from inference_pipeline import InferencePipeline
    from sonar_guard import SonarGuard
    from state_machine import NavStateMachine

    bot      = OmniBot()
    pipeline = InferencePipeline()
    pipeline.start()
    sonar    = SonarGuard()
    sonar.start()

    print("Waiting for first depth frame …")
    while True:
        _, depth, _ = pipeline.get_state()
        if depth is not None:
            break
        time.sleep(0.1)
    print("Hardware ready.\n")

    nsm = NavStateMachine(bot, pipeline, sonar=sonar, skip_scan=not args.scan)

    _ensure_csv()
    trial_offset = _count_existing_trials()
    session_rows: list[dict] = []

    # Handle Ctrl-C at the top level (between trials)
    def _sigint(sig, frame):
        print("\n\nInterrupted — writing partial results.")
        _print_summary(session_rows)
        nsm.stop()
        sonar.stop()
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    for i in range(args.trials):
        trial_no = trial_offset + i + 1
        row = run_trial(nsm, trial_no, do_scan=args.scan, timeout_s=effective_timeout)
        if not row:
            continue
        _append_row(row)
        session_rows.append(row)
        print(f"\n  Row written to {CSV_PATH}")
        if i < args.trials - 1:
            cont = input("\n  Press Enter for next trial (or q to quit): ").strip().lower()
            if cont == "q":
                break

    _print_summary(session_rows)
    nsm.stop()
    sonar.stop()
    pipeline.stop()


if __name__ == "__main__":
    main()
