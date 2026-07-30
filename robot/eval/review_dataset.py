"""
review_dataset.py — Manual review/correction tool for YOLO-format datasets
produced by capture_dataset.py.

capture_dataset.py saves the *model's own* detections as ground truth, which
is fine for a quick capture but not valid for evaluation until a human has
verified/corrected every box — otherwise eval_detection.py just measures the
model against itself. This tool lets you walk through the captured images,
delete false positives, draw missed detections, and fix wrong boxes/classes.

Usage
-----
    python review_dataset.py --dir dataset

Controls
--------
    Left-drag        Draw a new box, then press a class number (0-8) to confirm
    Right-click       Delete the box under the cursor
    S / Enter         Save this image's annotations and go to the next one
    N                 Skip to the next image WITHOUT saving changes
    P                 Go to the previous image (reloads from disk, no autosave)
    0-8               Assign a class to the box just drawn (see legend in HUD)
    Q / ESC           Quit (does NOT autosave the current image — press S first)
"""

import argparse
from pathlib import Path

import cv2

from inference_pipeline import BEVERAGE_LABELS

_BOX_COLOR      = (0, 255, 80)     # confirmed box - green
_PENDING_COLOR  = (0, 220, 255)    # box awaiting a class key - yellow
_FONT           = cv2.FONT_HERSHEY_SIMPLEX


class Box:
    __slots__ = ("cls", "x1", "y1", "x2", "y2")

    def __init__(self, cls, x1, y1, x2, y2):
        self.cls, self.x1, self.y1, self.x2, self.y2 = cls, x1, y1, x2, y2


def load_boxes(txt_path: Path, w: int, h: int) -> list[Box]:
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append(Box(cls, x1, y1, x2, y2))
    return boxes


def save_boxes(txt_path: Path, boxes: list[Box], w: int, h: int) -> None:
    lines = []
    for b in boxes:
        x1, x2 = sorted((b.x1, b.x2))
        y1, y2 = sorted((b.y1, b.y2))
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines))


def point_in_box(x, y, b: Box) -> bool:
    x1, x2 = sorted((b.x1, b.x2))
    y1, y2 = sorted((b.y1, b.y2))
    return x1 <= x <= x2 and y1 <= y <= y2


class Reviewer:
    def __init__(self, directory: Path):
        self.dir = directory
        self.images = sorted(p for p in directory.glob("*.jpg"))
        if not self.images:
            raise SystemExit(f"No .jpg files found in {directory}")
        self.idx = 0
        self.boxes: list[Box] = []
        self.pending: Box | None = None
        self.drag_start = None
        self.dirty = False
        self._load_current()

    # ── image loading ────────────────────────────────────────────────────────

    def _load_current(self):
        self.img = cv2.imread(str(self.images[self.idx]))
        self.h, self.w = self.img.shape[:2]
        txt_path = self.images[self.idx].with_suffix(".txt")
        self.boxes = load_boxes(txt_path, self.w, self.h)
        self.pending = None
        self.dirty = False

    def save(self):
        txt_path = self.images[self.idx].with_suffix(".txt")
        save_boxes(txt_path, self.boxes, self.w, self.h)
        self.dirty = False
        print(f"Saved {txt_path.name} ({len(self.boxes)} boxes)")

    def next(self, save_first: bool):
        if save_first:
            self.save()
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._load_current()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_current()

    # ── mouse handling ────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.drag_start = None
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                self.pending = Box(-1, x0, y0, x, y)
                print("Box drawn — press 0-8 to assign a class "
                      "(see legend), or draw again to replace it.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            for b in reversed(self.boxes):
                if point_in_box(x, y, b):
                    self.boxes.remove(b)
                    self.dirty = True
                    print("Deleted a box.")
                    break

    # ── drawing ───────────────────────────────────────────────────────────────

    def render(self):
        vis = self.img.copy()
        for b in self.boxes:
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            label = BEVERAGE_LABELS[b.cls] if 0 <= b.cls < len(BEVERAGE_LABELS) else "?"
            cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOR, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)), _FONT, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)

        if self.pending:
            x1, y1, x2, y2 = int(self.pending.x1), int(self.pending.y1), int(self.pending.x2), int(self.pending.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), _PENDING_COLOR, 2)

        hud = [
            f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}"
            + ("  *unsaved*" if self.dirty else ""),
            "Left-drag=new box  Right-click=delete  0-8=class  S=save+next  N=skip  P=prev  Q=quit",
        ]
        for i, line in enumerate(hud):
            y = 20 + i * 22
            cv2.putText(vis, line, (10, y), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (10, y), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        legend = " | ".join(f"{i}:{lbl}" for i, lbl in enumerate(BEVERAGE_LABELS))
        cv2.putText(vis, legend, (10, self.h - 12), _FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, legend, (10, self.h - 12), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return vis


def main():
    parser = argparse.ArgumentParser(description="Manually review/correct a YOLO dataset")
    parser.add_argument("--dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    reviewer = Reviewer(args.dir)
    window = "review_dataset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 700)
    cv2.setMouseCallback(window, reviewer.on_mouse)

    print(f"{len(reviewer.images)} images loaded from {args.dir}")
    print("Class legend: " + " | ".join(f"{i}={lbl}" for i, lbl in enumerate(BEVERAGE_LABELS)))

    while True:
        cv2.imshow(window, reviewer.render())
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            if reviewer.dirty:
                print("Quitting WITHOUT saving unsaved changes on this image "
                      "(press S first if you want to keep them).")
            break

        elif key in (ord('s'), ord('S'), 13):  # S or Enter
            reviewer.next(save_first=True)

        elif key in (ord('n'), ord('N')):
            reviewer.next(save_first=False)

        elif key in (ord('p'), ord('P')):
            reviewer.prev()

        elif ord('0') <= key <= ord('8') and reviewer.pending:
            reviewer.pending.cls = key - ord('0')
            reviewer.boxes.append(reviewer.pending)
            reviewer.pending = None
            reviewer.dirty = True

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
