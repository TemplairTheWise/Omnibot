"""
find_worst_boxes.py — Finds the N ground-truth boxes whose best-matching
prediction (by IoU, regardless of predicted class) differs from it the most,
and saves side-by-side visualisations for qualitative analysis.

Only GT boxes that have at least one prediction in the same image are
considered (so there is always something concrete to draw and compare).
"Most different" = lowest IoU between the GT box and its best-matching
prediction in that image.

Usage
-----
    python find_worst_boxes.py --images dataset --n 3 --out ../../resources
"""

import argparse
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

from eval_detection import _iou, _load_gt
from inference_pipeline import BEVERAGE_LABELS, DET_CONF_THRESH, DET_HEF_PATH, _parse_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the most different GT/prediction box pairs")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("resources"))
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    exclude_ids = {BEVERAGE_LABELS.index(l) for l in args.exclude}
    args.out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    hef = HEF(str(DET_HEF_PATH))
    candidates = []  # (iou, img_path, gt_box, pred_box, pred_label, pred_conf)

    with VDevice() as device:
        network_group = device.configure(
            hef, ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        )[0]
        in_info = hef.get_input_vstream_infos()[0]
        det_h, det_w = in_info.shape[0], in_info.shape[1]
        in_params  = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        ng_params  = network_group.create_params()

        with InferVStreams(network_group, in_params, out_params) as pipe:
            for img_path in image_paths:
                gt_boxes = _load_gt(img_path.with_suffix(".txt"))
                gt_boxes = [g for g in gt_boxes if g["cls"] not in exclude_ids]
                if not gt_boxes:
                    continue

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                inp = cv2.resize(rgb, (det_w, det_h))
                batch = np.expand_dims(inp, axis=0)

                with network_group.activate(ng_params):
                    outputs = pipe.infer({in_info.name: batch})
                preds = _parse_detections(outputs, BEVERAGE_LABELS, det_w, det_h)
                preds = [p for p in preds if BEVERAGE_LABELS.index(p.label) not in exclude_ids]
                if not preds:
                    continue  # nothing to compare this image's GT boxes against

                for g in gt_boxes:
                    best_iou, best_p = -1.0, None
                    for p in preds:
                        pred_box = {"x1": p.x1, "y1": p.y1, "x2": p.x2, "y2": p.y2}
                        iou = _iou(pred_box, g)
                        if iou > best_iou:
                            best_iou, best_p = iou, p
                    candidates.append((best_iou, img_path, dict(g), best_p))

    candidates.sort(key=lambda c: c[0])  # lowest IoU first = most different
    worst = candidates[: args.n]

    print(f"{len(candidates)} GT boxes had a comparable prediction in-frame. "
          f"Showing the {len(worst)} most different:")

    def _outlined(vis, text, pos, scale, color):
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    for rank, (iou, img_path, g, p) in enumerate(worst, start=1):
        img = cv2.imread(str(img_path))
        vis = img.copy()
        h_img, w_img = img.shape[:2]
        gt_label = BEVERAGE_LABELS[g["cls"]] if 0 <= g["cls"] < len(BEVERAGE_LABELS) else "?"

        # g/p coordinates are normalised [0,1] (eval_detection._load_gt and
        # _parse_detections both use that convention) - scale to pixels here.
        gx1, gy1, gx2, gy2 = (int(g["x1"] * w_img), int(g["y1"] * h_img),
                              int(g["x2"] * w_img), int(g["y2"] * h_img))
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        # GT label always above the box, clamped to stay on-frame.
        _outlined(vis, f"GT: {gt_label}", (gx1, max(gy1 - 10, 20)), 0.55, (0, 255, 0))

        px1, py1, px2, py2 = (int(p.x1 * w_img), int(p.y1 * h_img),
                              int(p.x2 * w_img), int(p.y2 * h_img))
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
        # Pred label always below the box, clamped to stay on-frame.
        _outlined(vis, f"Pred: {p.label} {p.confidence:.0%}",
                  (px1, min(py2 + 22, h_img - 10)), 0.55, (0, 0, 255))

        # IoU caption bottom-left, well clear of the GT/pred labels above.
        _outlined(vis, f"IoU = {iou:.3f}", (10, h_img - 12), 0.7, (255, 255, 255))

        out_path = args.out / f"worst_box_{rank}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  #{rank}  {img_path.name}  IoU={iou:.3f}  "
              f"GT={gt_label}  Pred={p.label}({p.confidence:.0%})  -> {out_path}")


if __name__ == "__main__":
    main()
