"""
eval_detection_agnostic.py — Class-agnostic mAP@0.5 evaluation.

Same dataset/methodology as eval_detection.py, but ignores class labels
entirely on both the ground-truth and the prediction side - a match only
requires IoU >= 0.5 with ANY unmatched ground-truth box, regardless of which
class either side claims. Answers "does the model see that there is *a*
beverage container, and roughly where?", separate from "does it also name
the right class?". Useful for telling apart localisation failures from
class-confusion failures when the per-class mAP looks unexpectedly low.

Usage
-----
    python eval_detection_agnostic.py --images dataset_harsh/

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

from eval_detection import _compute_ap, _iou, _load_gt, IOU_THRESHOLD
from inference_pipeline import BEVERAGE_LABELS, DET_CONF_THRESH, DET_HEF_PATH, _parse_detections

log = logging.getLogger(__name__)


def evaluate(images_dir: Path, conf_thresh: float, exclude_ids: frozenset[int] = frozenset()) -> dict:
    exclude_labels = {BEVERAGE_LABELS[i] for i in exclude_ids}
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
            ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe),
        )[0]

        in_info  = hef.get_input_vstream_infos()[0]
        det_h, det_w = in_info.shape[0], in_info.shape[1]

        in_params  = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        ng_params  = network_group.create_params()

        all_preds: list[tuple[float, int]] = []  # (confidence, is_tp) - single pseudo-class
        n_gt_total = 0
        latencies: list[float] = []
        skipped = 0

        with InferVStreams(network_group, in_params, out_params) as pipe:
            log.info("Running inference on %d images (input %dx%d) — class-agnostic …",
                     len(image_paths), det_w, det_h)

            for img_path in image_paths:
                txt_path = img_path.with_suffix(".txt")
                gt_boxes = _load_gt(txt_path)
                if exclude_ids:
                    gt_boxes = [g for g in gt_boxes if g["cls"] not in exclude_ids]
                if not gt_boxes:
                    skipped += 1
                    continue
                n_gt_total += len(gt_boxes)

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                inp = cv2.resize(rgb, (det_w, det_h))
                batch = np.expand_dims(inp, axis=0)

                t0 = time.perf_counter()
                with network_group.activate(ng_params):
                    outputs = pipe.infer({in_info.name: batch})
                latencies.append(time.perf_counter() - t0)

                preds = _parse_detections(outputs, BEVERAGE_LABELS, det_w, det_h)
                if exclude_labels:
                    preds = [d for d in preds if d.label not in exclude_labels]

                # Class-agnostic greedy matching: any prediction may match any
                # unmatched GT box, purely on IoU. No class_id involved at all.
                for det in sorted(preds, key=lambda d: -d.confidence):
                    pred_box = {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2}
                    best_iou, best_i = 0.0, -1
                    for i, g in enumerate(gt_boxes):
                        if g["matched"]:
                            continue
                        iou = _iou(pred_box, g)
                        if iou > best_iou:
                            best_iou, best_i = iou, i
                    if best_iou >= IOU_THRESHOLD and best_i >= 0:
                        gt_boxes[best_i]["matched"] = True
                        all_preds.append((det.confidence, 1))
                    else:
                        all_preds.append((det.confidence, 0))

    preds_sorted = sorted(all_preds, key=lambda x: -x[0])
    tps, fps = 0, 0
    recs, precs = [], []
    for _, is_tp in preds_sorted:
        if is_tp:
            tps += 1
        else:
            fps += 1
        recs.append(tps / n_gt_total if n_gt_total else 0.0)
        precs.append(tps / (tps + fps) if (tps + fps) > 0 else 0.0)

    ap        = _compute_ap(recs, precs) if preds_sorted else 0.0
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall    = tps / n_gt_total if n_gt_total else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    lat_ms = np.array(latencies) * 1000
    return {
        "ap50_class_agnostic": round(ap, 4),
        "precision":           round(precision, 4),
        "recall":              round(recall, 4),
        "f1":                  round(f1, 4),
        "tp":                  tps,
        "fp":                  fps,
        "gt":                  n_gt_total,
        "n_images":            len(image_paths) - skipped,
        "n_skipped":           skipped,
        "latency_mean_ms":     round(float(lat_ms.mean()), 1) if len(lat_ms) else None,
        "conf_thresh":         conf_thresh,
        "iou_thresh":          IOU_THRESHOLD,
        "excluded_classes":    sorted(exclude_labels),
    }


def _print_result(r: dict) -> None:
    W = 60
    print(f"\n{'─'*W}")
    print("  CLASS-AGNOSTIC evaluation (presence + localisation only)")
    if r.get("excluded_classes"):
        print(f"  Excluded from evaluation: {', '.join(r['excluded_classes'])}")
    print(f"{'─'*W}")
    print(f"  AP@0.5     = {r['ap50_class_agnostic']:.4f}")
    print(f"  Precision  = {r['precision']:.4f}")
    print(f"  Recall     = {r['recall']:.4f}")
    print(f"  F1         = {r['f1']:.4f}")
    print(f"  TP/GT      = {r['tp']}/{r['gt']}   FP = {r['fp']}")
    print(f"  Images     = {r['n_images']} ({r['n_skipped']} skipped)")
    if r["latency_mean_ms"] is not None:
        print(f"  Latency    = {r['latency_mean_ms']:.1f} ms mean")
    print(f"{'─'*W}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Class-agnostic mAP@0.5 evaluation")
    parser.add_argument("--images", type=Path, required=True, metavar="DIR")
    parser.add_argument("--conf", type=float, default=DET_CONF_THRESH, metavar="F")
    parser.add_argument("--out", type=Path, default=Path("eval_detection_agnostic_results.json"))
    parser.add_argument("--exclude", action="append", default=[], metavar="LABEL",
                        help="Class label to exclude entirely from evaluation (both GT and "
                             "predictions), e.g. --exclude cup-disposable. Repeatable.")
    args = parser.parse_args()

    exclude_ids = set()
    for label in args.exclude:
        if label not in BEVERAGE_LABELS:
            raise SystemExit(f"Unknown --exclude label {label!r}. Valid labels: {BEVERAGE_LABELS}")
        exclude_ids.add(BEVERAGE_LABELS.index(label))

    result = evaluate(args.images.expanduser().resolve(), args.conf, exclude_ids=frozenset(exclude_ids))
    _print_result(result)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Results saved → {args.out}")


if __name__ == "__main__":
    main()
