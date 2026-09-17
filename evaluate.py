import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from dataset import DEFAULT_DATA, list_images, read_pair
from metrics import confusion_matrix, scores
from model import choose_device, load_checkpoint
from predict import overlay_mask, predict_mask, write_image


def main():
    # set up directories and load the model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("examples must be nonnegative")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    # choose the evaluation images
    if args.split == "val":
        folder = "train"
        names = checkpoint["split"]["val"]
    else:
        folder = "test"
        names = list_images(args.data, "test")
    if not names:
        raise ValueError("Evaluation split is empty")
    args.out.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((2, 2), dtype=np.int64)
    # predicting other everywhere can have deceptively high accuracy, include it as a baseline.
    baseline_matrix = np.zeros_like(matrix)
    rows, elapsed = [], 0.0
    first, _ = read_pair(args.data, folder, names[0])
    # Predict and score images
    for _ in range(3):
        predict_mask(model, first, checkpoint["size"], device)
    for index, name in enumerate(names):
        image, target = read_pair(args.data, folder, name)
        start = time.perf_counter()
        prediction = predict_mask(model, image, checkpoint["size"], device)
        elapsed += time.perf_counter() - start
        image_matrix = confusion_matrix(prediction, target)
        # Aggregate pixel counts first
        matrix += image_matrix
        baseline_matrix += confusion_matrix(np.zeros_like(target), target)
        result = scores(image_matrix)
        rows.append({"image": name, "true_trail_pixels": int((target == 1).sum()),
                     "predicted_trail_pixels": int(((prediction == 1) & (target != 255)).sum()),
                     **{k: result[k] for k in
                     ("trail_iou", "trail_precision", "trail_recall", "valid_pixels")}})
        if (index + 1) % 25 == 0:
            print(f"Evaluated {index + 1}/{len(names)}", flush=True)

    # save the overall result
    report = {"split": args.split, "images": len(names), "checkpoint": str(args.checkpoint),
              "checkpoint_epoch": checkpoint["epoch"], "input_size": checkpoint["size"],
              "scoring_resolution": "original annotation resolution",
              "metrics": scores(matrix), "all_other_baseline": scores(baseline_matrix),
              "prediction_fps": len(names) / elapsed, "device": str(device),
              "timing_scope": "preprocessing + transfer + model + resize + CPU mask; excludes disk IO, metrics and drawing",
              "split_method": checkpoint["split"]["method"]}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Export high/low IoU examples for inspection
    ranked = sorted((row for row in rows if row["trail_iou"] is not None), key=lambda r: r["trail_iou"])
    selections = {"worst": ranked[:args.examples],
                  "best": list(reversed(ranked))[:args.examples]}
    for category, selected in selections.items():
        for rank, row in enumerate(selected, 1):
            image, target = read_pair(args.data, folder, row["image"])
            prediction = predict_mask(model, image, checkpoint["size"], device)
            panels = [overlay_mask(image, np.zeros_like(target), "Input"),
                      overlay_mask(image, target, "Ground truth (purple = ignore)"),
                      overlay_mask(image, prediction, f"Prediction | IoU {row['trail_iou']:.3f}")]
            write_image(args.out / category / f"{rank:02d}_{row['image']}", cv2.hconcat(panels))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
