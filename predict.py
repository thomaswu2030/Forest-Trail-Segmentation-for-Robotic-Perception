import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from dataset import image_to_tensor
from model import choose_device, load_checkpoint


# predict a label for each pixel
@torch.inference_mode()
def predict_mask(model, image, size, device):
    # add a batch dimension - [3, H, W] becomes [1, 3, H, W].
    tensor = image_to_tensor(image, size).unsqueeze(0).to(device)
    logits = model(tensor)["out"]
    # resize scores, then choose a category
    logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    # choose the highest-scoring class, remove the batch dimension, and return a CPU mask
    return logits.argmax(1)[0].byte().cpu().numpy()


# draw a prediction overlay
def overlay_mask(image, labels, title="Predicted trail"):
    output = image.copy()
    selected = labels == 1
    output[selected] = (0.55 * image[selected] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
    output[labels == 255] = (180, 0, 180) 
    if title:
        cv2.rectangle(output, (0, 0), (output.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(output, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return output


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


# video processing
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--show", action="store_true", help="Optional window; Q exits video")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = entire video")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input file does not exist")
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    if args.max_frames < 0:
        parser.error("max-frames must be nonnegative")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    # reuse training dimensions so callers do not accidentally change preprocessing
    size = checkpoint["size"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.input.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        mask_output = args.output.with_name(args.output.stem + "_mask.png")
        if mask_output.resolve() == args.input.resolve():
            parser.error("Generated mask filename would overwrite the input image")
        image = cv2.imread(str(args.input))
        if image is None:
            raise ValueError("OpenCV could not decode the input image")
        labels = predict_mask(model, image, size, device)
        overlay = overlay_mask(image, labels)
        write_image(args.output, overlay)
        # export other as black and trail as white, this is not the training ignore label
        write_image(mask_output, labels * 255)
        if args.show:
            cv2.imshow("Predicted trail", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Saved overlay and binary mask to {args.output.parent}")
        return

    if args.output.suffix.lower() != ".mp4":
        parser.error("Use an .mp4 output for video input")
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    count, inference_seconds = 0, 0.0
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not create the MP4 writer")
                predict_mask(model, frame, size, device)  # Untimed warm-up.
            start = time.perf_counter()
            labels = predict_mask(model, frame, size, device)
            inference_seconds += time.perf_counter() - start
            overlay = overlay_mask(frame, labels)
            writer.write(overlay)
            count += 1
            if count % 30 == 0:
                print(f"Processed {count} frames", flush=True)
            if args.show:
                cv2.imshow("Predicted trail (Q to quit)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and count >= args.max_frames:
                break
        if count == 0:
            raise ValueError("Video contains no decodable frames")
    # release video handles even when decoding or prediction raises an exception
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    report = {"frames": count, "input_video_fps": fps,
              "prediction_fps": count / inference_seconds,
              "timing_scope": "preprocessing + transfer + model + resize + CPU mask; excludes video IO and drawing",
              "checkpoint": str(args.checkpoint), "size": size, "device": str(device)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
