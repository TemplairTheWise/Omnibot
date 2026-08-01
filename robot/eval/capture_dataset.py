"""
capture_dataset.py — Interactive YOLO dataset capture.

Shows a live camera preview with detection bounding boxes overlaid.
Press SPACE to save the current frame as a JPEG + matching YOLO .txt
annotation. Press Q to quit (ESC is intentionally not bound - see the
in-loop comment on this Pi's Wayland/XWayland phantom-ESC issue).

Usage
-----
    python capture_dataset.py [--out DIR] [--no-flip] [--conf 0.25]

    --out     DIR   Output directory (default: dataset/)
    --flip          Rotate preview 180° for viewing (default: ON, since the
                    camera is physically mounted rotated; saved images are
                    NOT flipped — they match the coordinate space the model
                    was trained in)
    --no-flip       Disable the 180° preview rotation
    --conf  FLOAT   Minimum confidence to include a detection (default: 0.25)

YOLO annotation format (one line per detection):
    class_id  cx  cy  w  h     (all coordinates normalised [0, 1])

Pressing SPACE with no detections saves an image + an empty .txt (negative
example — valid for mAP evaluation and model fine-tuning).

Output layout
-------------
    dataset/
        0001.jpg
        0001.txt
        0002.jpg
        0002.txt
        ...

Requirements
-----------
  • Hailo hardware connected (for detection)
  • X display or VNC session for the cv2 preview window
  • Source setup_env.sh first
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from inference_pipeline import (
    BEVERAGE_LABELS,
    DET_CONF_THRESH,
    InferencePipeline,
)

log = logging.getLogger(__name__)

# ── Visual style ──────────────────────────────────────────────────────────────
_BOX_COLOR  = (0, 255, 80)    # BGR green
_TEXT_COLOR = (0, 255, 80)
_DIM_COLOR  = (60, 120, 60)
_SAVE_COLOR = (255, 255, 255)
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_SAVED_FLASH_FRAMES = 8       # how many frames to show "SAVED" banner


def _draw_overlay(frame: np.ndarray, detections, saved_count: int,
                  flash: int, conf_thresh: float) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Detection boxes ───────────────────────────────────────────────────────
    for det in detections:
        if det.confidence < conf_thresh:
            continue
        x1 = int(det.x1 * w)
        y1 = int(det.y1 * h)
        x2 = int(det.x2 * w)
        y2 = int(det.y2 * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"{det.label}  {det.confidence:.0%}"
        ts    = cv2.getTextSize(label, _FONT, 0.52, 1)[0]
        # label background
        cv2.rectangle(vis, (x1, y1 - ts[1] - 6), (x1 + ts[0] + 6, y1), _BOX_COLOR, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), _FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    # ── HUD — top-left ────────────────────────────────────────────────────────
    n_det = sum(1 for d in detections if d.confidence >= conf_thresh)
    hud_lines = [
        f"Detections : {n_det}",
        f"Saved      : {saved_count}",
        "SPACE = save   Q/ESC = quit",
    ]
    for i, line in enumerate(hud_lines):
        y = 20 + i * 22
        cv2.putText(vis, line, (10, y), _FONT, 0.55, (0, 0, 0),    3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y), _FONT, 0.55, _TEXT_COLOR,   1, cv2.LINE_AA)

    # ── "SAVED" flash banner ──────────────────────────────────────────────────
    if flash > 0:
        text = f"SAVED  #{saved_count}"
        ts   = cv2.getTextSize(text, _FONT, 1.4, 3)[0]
        cx   = (w - ts[0]) // 2
        cy   = h // 2 + ts[1] // 2
        cv2.putText(vis, text, (cx, cy), _FONT, 1.4, (0, 0, 0),     4, cv2.LINE_AA)
        cv2.putText(vis, text, (cx, cy), _FONT, 1.4, _SAVE_COLOR,    2, cv2.LINE_AA)

    return vis


def _save_sample(out_dir: Path, idx: int,
                 frame: np.ndarray,
                 detections, conf_thresh: float) -> None:
    stem = f"{idx:04d}"
    # Save image
    img_path = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save YOLO annotation
    txt_path = out_dir / f"{stem}.txt"
    lines    = []
    for det in detections:
        if det.confidence < conf_thresh:
            continue
        try:
            cls_id = BEVERAGE_LABELS.index(det.label)
        except ValueError:
            continue
        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        bw = det.x2 - det.x1
        bh = det.y2 - det.y1
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines))
    log.info("Saved %s  (%d detections)", stem, len(lines))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Interactive YOLO dataset capture")
    parser.add_argument("--out",  type=Path, default=Path("dataset"),
                        help="Output directory (default: dataset/)")
    parser.add_argument("--flip", dest="flip", action="store_true", default=True,
                        help="Rotate preview 180° (default: on - the camera is "
                             "physically mounted rotated; does not affect saved images)")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        help="Disable the 180° preview rotation")
    parser.add_argument("--conf", type=float, default=DET_CONF_THRESH,
                        help=f"Min detection confidence (default {DET_CONF_THRESH})")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Count existing samples so we resume numbering correctly
    existing = sorted(args.out.glob("*.jpg"))
    idx = int(existing[-1].stem) + 1 if existing else 1
    if existing:
        log.info("Resuming — %d existing samples found, next index %d",
                 len(existing), idx)

    print(f"\nStarting camera + detection …")
    print(f"Output → {args.out.resolve()}")
    print(f"Conf threshold = {args.conf}")
    print("─" * 40)

    pipeline = InferencePipeline()
    pipeline.start()

    # Wait for first frame
    print("Waiting for camera …", end="", flush=True)
    while True:
        frame, _, _ = pipeline.get_state()
        if frame is not None:
            break
        time.sleep(0.05)
    print(" ready.\n")

    cv2.namedWindow("capture_dataset", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("capture_dataset", 800, 600)

    saved_count = idx - 1
    flash       = 0

    try:
        while True:
            frame, _, detections = pipeline.get_state()
            if frame is None:
                time.sleep(0.02)
                continue

            # Preview frame (optionally flipped for viewing)
            preview = cv2.rotate(frame, cv2.ROTATE_180) if args.flip else frame

            # Mirror detections for display if flipped
            display_dets = detections
            if args.flip:
                mirrored = []
                for d in detections:
                    mirrored.append(d._replace(
                        x1=1.0 - d.x2, x2=1.0 - d.x1,
                        y1=1.0 - d.y2, y2=1.0 - d.y1,
                    ))
                display_dets = mirrored

            vis = _draw_overlay(preview, display_dets, saved_count, flash, args.conf)
            if flash > 0:
                flash -= 1

            cv2.imshow("capture_dataset", vis)
            key = cv2.waitKey(30) & 0xFF

            # NOTE: ESC (27) is deliberately NOT treated as quit here - on this
            # Pi's Wayland/XWayland setup, a freshly mapped window that doesn't
            # get real focus can synthesize a sustained phantom ESC, which would
            # otherwise close the window before the user ever touches a key.
            # Use 'q' or the window's close button instead.
            if key in (ord('q'), ord('Q')):
                break

            if key == ord(' '):
                _save_sample(args.out, idx, frame, detections, args.conf)
                saved_count += 1
                idx         += 1
                flash        = _SAVED_FLASH_FRAMES

            if cv2.getWindowProperty("capture_dataset", cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed via the OS close button

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

    print(f"\nDone — {saved_count} samples saved to {args.out.resolve()}")
    if saved_count > 0:
        print(f"Run evaluation:  python eval_detection.py --images {args.out}")


if __name__ == "__main__":
    main()
