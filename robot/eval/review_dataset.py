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
    python review_dataset.py --dir dataset [--no-flip]

Display is rotated 180° by default to match capture_dataset.py's preview
(the camera is physically mounted rotated) - this only affects what you see
on screen, never the underlying box coordinates written to the .txt files.

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
    def __init__(self, directory: Path, flip: bool = True):
        self.dir = directory
        self.images = sorted(p for p in directory.glob("*.jpg"))
        if not self.images:
            raise SystemExit(f"No .jpg files found in {directory}")
        self.idx = 0
        self.boxes: list[Box] = []
        self.pending: Box | None = None
        self.drag_start = None
        self.dirty = False
        # The saved images are in the camera's raw (upside-down) orientation -
        # same convention as capture_dataset.py. All box math below stays in
        # that raw coordinate space (it's what gets written to the .txt file);
        # only the display and mouse input are flipped for viewing.
        self.flip = flip
        self._load_current()

    # ── raw <-> display coordinate conversion (180° rotation) ───────────────────

    def _to_raw(self, x, y):
        if not self.flip:
            return x, y
        return self.w - x, self.h - y

    def _to_display_box(self, x1, y1, x2, y2):
        if not self.flip:
            return x1, y1, x2, y2
        return self.w - x2, self.h - y2, self.w - x1, self.h - y1

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
        x, y = self._to_raw(x, y)  # screen click -> raw storage coordinates
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

    @staticmethod
    def _put_text_outlined(vis, text, pos, scale, color):
        cv2.putText(vis, text, pos, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, pos, _FONT, scale, color, 1, cv2.LINE_AA)

    def _wrap_to_width(self, items, max_width, scale):
        """Greedily pack ' | '-separated items into lines that fit max_width px."""
        lines, current = [], ""
        for item in items:
            candidate = item if not current else f"{current} | {item}"
            (tw, _), _ = cv2.getTextSize(candidate, _FONT, scale, 1)
            if tw > max_width and current:
                lines.append(current)
                current = item
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def render(self):
        # Rotate the base frame for display; box overlays are transformed to
        # match, but their underlying (raw) coordinates are never touched.
        vis = cv2.rotate(self.img, cv2.ROTATE_180).copy() if self.flip else self.img.copy()

        for b in self.boxes:
            x1, y1, x2, y2 = self._to_display_box(b.x1, b.y1, b.x2, b.y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = BEVERAGE_LABELS[b.cls] if 0 <= b.cls < len(BEVERAGE_LABELS) else "?"
            cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOR, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)), _FONT, 0.5, _BOX_COLOR, 1, cv2.LINE_AA)

        if self.pending:
            x1, y1, x2, y2 = self._to_display_box(
                self.pending.x1, self.pending.y1, self.pending.x2, self.pending.y2
            )
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), _PENDING_COLOR, 2)

        hud = [
            f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}"
            + ("  *unsaved*" if self.dirty else ""),
            "Left-drag=new box  Right-click=delete  0-8=class  S=save+next  N=skip  P=prev  Q=quit",
        ]
        for i, line in enumerate(hud):
            self._put_text_outlined(vis, line, (10, 20 + i * 22), 0.5, (255, 255, 255))

        # Legend, wrapped to fit the frame width instead of running off the edge.
        legend_scale = 0.45
        items = [f"{i}:{lbl}" for i, lbl in enumerate(BEVERAGE_LABELS)]
        legend_lines = self._wrap_to_width(items, max_width=self.w - 20, scale=legend_scale)
        base_y = self.h - 12 - 20 * (len(legend_lines) - 1)
        for i, line in enumerate(legend_lines):
            self._put_text_outlined(vis, line, (10, base_y + i * 20), legend_scale, (255, 255, 255))

        return vis


def main():
    parser = argparse.ArgumentParser(description="Manually review/correct a YOLO dataset")
    parser.add_argument("--dir", type=Path, default=Path("dataset"))
    parser.add_argument("--flip", dest="flip", action="store_true", default=True,
                        help="Rotate the display 180° (default: on - matches "
                             "capture_dataset.py's preview; does not affect saved files)")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        help="Disable the 180° display rotation")
    args = parser.parse_args()

    reviewer = Reviewer(args.dir, flip=args.flip)
    window = "review_dataset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 700)
    cv2.setMouseCallback(window, reviewer.on_mouse)

    print(f"{len(reviewer.images)} images loaded from {args.dir}")
    print("Class legend: " + " | ".join(f"{i}={lbl}" for i, lbl in enumerate(BEVERAGE_LABELS)))

    while True:
        cv2.imshow(window, reviewer.render())
        key = cv2.waitKey(30) & 0xFF

        # NOTE: ESC (27) deliberately not treated as quit - see capture_dataset.py
        if key in (ord('q'), ord('Q')):
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
