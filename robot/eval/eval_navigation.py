"""
eval_navigation.py — Step 14: Interactive navigation trial runner.

Run this script with the robot physically present.  For each trial it:
  1. Prompts you for the scenario name and expected target.
  2. Starts the NavStateMachine (with or without polar scan).
  3. Waits until the robot reaches FOUND or you abort with Ctrl-C.
  4. Prompts you to confirm success and note any observations.
  5. Writes one row to eval_navigation_results.csv.

After all trials it prints a summary table.

Usage
-----
    python eval_navigation.py [--trials N] [--scan] [--timeout 120]

Output
------
  eval_navigation_results.csv  (appended, so runs accumulate across sessions)

Requirements
-----------
  • Hailo hardware + motors connected
  • Source setup_env.sh first
"""

import argparse
import csv
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

CSV_PATH = Path("eval_navigation_results.csv")
FIELDNAMES = [
    "trial", "scenario", "target", "success",
    "outcome", "duration_s", "detections_count", "session_id", "notes",
]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


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
    record   = input("  Record video path (blank = off): ").strip() or None

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

    # Grab detections count + session id from the last session log
    history = nsm.get_history(limit=1)
    det_count  = history[0]["detections_count"] if history else 0
    session_id = history[0]["session_id"] if history else None

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
        "duration_s":       round(elapsed, 1),
        "detections_count": det_count,
        "session_id":       session_id or "",
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
                        help="Per-trial timeout in seconds (default 120)")
    args = parser.parse_args()

    # Hardware init
    print("\nInitialising hardware …")
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
        row = run_trial(nsm, trial_no, do_scan=args.scan, timeout_s=args.timeout)
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
