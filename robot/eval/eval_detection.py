"""
eval_detection.py — Step 13: mAP@0.5 evaluation for the beverage detector.

Runs the YOLO model on a directory of images that have matching YOLO-format
ground-truth .txt annotation files, then reports per-class AP, overall mAP@0.5,
precision, recall, F1, and per-image latency.

Usage
-----
    python eval_detection.py --images /path/to/images/

Image directory layout
-----------------------
    images/
      foo.jpg          <- image
      foo.txt          <- YOLO annotation: one line per object
                          "class_id cx cy w h"  (coords normalised [0,1])
      bar.png
      bar.txt
      …

Output
------
  Printed table + eval_detection_results.json in the current directory.

Requirements
-----------
  • Hailo hardware connected (loads yolo26_split.hef via inference_pipeline constants)
  • Source setup_env.sh first (virtual env + PYTHONPATH)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
    OutputVStreamParams,
    VDevice,
)

from inference_pipeline import (
    BEVERAGE_LABELS,
    DET_CONF_THRESH,
    DET_HEF_PATH,
    NMS_IOU_THRESH,
    _parse_detections,
)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.50   # Pascal VOC mAP@0.5


# ── Ground-truth loading ───────────────────────────────────────────────────────

def _load_gt(txt_path: Path) -> list[dict]:
    """Parse a YOLO .txt annotation file into a list of GT boxes."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append({
            "cls": cls_id,
            "x1": cx - w / 2,
            "y1": cy - h / 2,
            "x2": cx + w / 2,
            "y2": cy + h / 2,
            "matched": False,
        })
    return boxes


# ── IoU ────────────────────────────────────────────────────────────────────────

def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (ua + ub - inter)


# ── Pascal VOC all-point interpolated AP ──────────────────────────────────────

def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """PASCAL VOC 2010+ all-point interpolation."""
    recalls    = [0.0] + list(recalls)    + [1.0]
    precisions = [1.0] + list(precisions) + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(images_dir: Path, conf_thresh: float) -> dict:
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    log.info("Loading model from %s …", DET_HEF_PATH)
    hef = HEF(str(DET_HEF_PATH))

    with VDevice() as device:
        network_group = device.configure(
            hef,
            ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe
            ),
        )[0]

        in_info  = hef.get_input_vstream_infos()[0]
        det_h, det_w = in_info.shape[0], in_info.shape[1]

        in_params  = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        ng_params  = network_group.create_params()

        # Per-class: list of (confidence, is_tp), plus GT count
        n_classes   = len(BEVERAGE_LABELS)
        class_preds: list[list[tuple[float, int]]] = [[] for _ in range(n_classes)]
        class_gt:    list[int]                     = [0] * n_classes
        latencies:   list[float]                   = []
        skipped      = 0

        with InferVStreams(network_group, in_params, out_params) as pipe:
            log.info("Running inference on %d images (input %dx%d) …",
                     len(image_paths), det_w, det_h)

            for img_path in image_paths:
                txt_path = img_path.with_suffix(".txt")
                gt_boxes = _load_gt(txt_path)
                if not gt_boxes:
                    log.debug("No annotation for %s — skipping", img_path.name)
                    skipped += 1
                    continue

                for g in gt_boxes:
                    if 0 <= g["cls"] < n_classes:
                        class_gt[g["cls"]] += 1

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    log.warning("Cannot read %s — skipping", img_path.name)
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                inp = cv2.resize(rgb, (det_w, det_h))          # uint8 (0-255)
                batch = np.expand_dims(inp, axis=0)             # (1, H, W, 3)

                t0 = time.perf_counter()
                with network_group.activate(ng_params):
                    outputs = pipe.infer({in_info.name: batch})
                latencies.append(time.perf_counter() - t0)

                preds = _parse_detections(outputs, BEVERAGE_LABELS, det_w, det_h)

                # Match predictions → GT (greedy by confidence)
                for det in sorted(preds, key=lambda d: -d.confidence):
                    try:
                        cls_id = BEVERAGE_LABELS.index(det.label)
                    except ValueError:
                        continue
                    pred_box = {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2}
                    best_iou, best_i = 0.0, -1
                    for i, g in enumerate(gt_boxes):
                        if g["matched"] or g["cls"] != cls_id:
                            continue
                        iou = _iou(pred_box, g)
                        if iou > best_iou:
                            best_iou, best_i = iou, i
                    if best_iou >= IOU_THRESHOLD and best_i >= 0:
                        gt_boxes[best_i]["matched"] = True
                        class_preds[cls_id].append((det.confidence, 1))
                    else:
                        class_preds[cls_id].append((det.confidence, 0))

    # ── Per-class AP ──────────────────────────────────────────────────────────
    per_class: list[dict] = []
    aps: list[float] = []

    for cls_id, label in enumerate(BEVERAGE_LABELS):
        n_gt      = class_gt[cls_id]
        preds_cls = sorted(class_preds[cls_id], key=lambda x: -x[0])

        if n_gt == 0 and not preds_cls:
            continue

        if n_gt == 0 or not preds_cls:
            per_class.append({"label": label, "ap": 0.0, "precision": 0.0,
                               "recall": 0.0, "f1": 0.0, "gt": n_gt,
                               "tp": 0, "fp": len(preds_cls)})
            aps.append(0.0)
            continue

        tps, fps = 0, 0
        recs, precs = [], []
        for _, is_tp in preds_cls:
            if is_tp:
                tps += 1
            else:
                fps += 1
            rec  = tps / n_gt
            prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            recs.append(rec)
            precs.append(prec)

        ap        = _compute_ap(recs, precs)
        precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        recall    = tps / n_gt
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class.append({
            "label":     label,
            "ap":        round(ap, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "gt":        n_gt,
            "tp":        tps,
            "fp":        fps,
        })
        aps.append(ap)

    lat_ms  = np.array(latencies) * 1000
    return {
        "map50":           round(float(np.mean(aps)) if aps else 0.0, 4),
        "n_images":        len(image_paths) - skipped,
        "n_skipped":       skipped,
        "latency_mean_ms": round(float(lat_ms.mean()), 1) if len(lat_ms) else None,
        "latency_p95_ms":  round(float(np.percentile(lat_ms, 95)), 1) if len(lat_ms) else None,
        "conf_thresh":     conf_thresh,
        "iou_thresh":      IOU_THRESHOLD,
        "per_class":       per_class,
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_table(results: dict) -> None:
    W = 68
    print(f"\n{'─'*W}")
    print(f"  mAP@0.5 = {results['map50']:.4f}  "
          f"({results['n_images']} images, {results['n_skipped']} skipped)")
    if results["latency_mean_ms"] is not None:
        print(f"  Latency  = {results['latency_mean_ms']:.1f} ms mean  "
              f"/ {results['latency_p95_ms']:.1f} ms p95")
    print(f"{'─'*W}")
    fmt = "  {:<22}  {:>6}  {:>6}  {:>6}  {:>6}  {:>5}/{}"
    print(fmt.format("Class", "AP", "Prec", "Rec", "F1", "TP", "GT"))
    print(f"{'─'*W}")
    for c in results["per_class"]:
        print(fmt.format(
            c["label"],
            f"{c['ap']:.4f}",
            f"{c['precision']:.4f}",
            f"{c['recall']:.4f}",
            f"{c['f1']:.4f}",
            c["tp"],
            c["gt"],
        ))
    print(f"{'─'*W}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="mAP@0.5 evaluation for yolo26_split.hef")
    parser.add_argument("--images", type=Path, required=True,
                        metavar="DIR", help="Directory with images + YOLO .txt annotations")
    parser.add_argument("--conf",   type=float, default=DET_CONF_THRESH,
                        metavar="F", help=f"Confidence threshold (default {DET_CONF_THRESH})")
    parser.add_argument("--out",    type=Path, default=Path("eval_detection_results.json"),
                        metavar="FILE", help="Output JSON (default eval_detection_results.json)")
    args = parser.parse_args()

    results = evaluate(
        images_dir  = args.images.expanduser().resolve(),
        conf_thresh = args.conf,
    )
    _print_table(results)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {args.out}")


if __name__ == "__main__":
    main()
