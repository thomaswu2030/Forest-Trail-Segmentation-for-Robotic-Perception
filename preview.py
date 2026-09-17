import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import DEFAULT_DATA, list_images, read_pair
from predict import overlay_mask, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/preview.jpg"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    names = list_images(args.data, args.split)
    if not 0 <= args.index < len(names):
        parser.error(f"index must be between 0 and {len(names)-1}")
    # this preview decodes human annotations
    image, target = read_pair(args.data, args.split, names[args.index])
    # unknown pixels start purple, other is dark gray and trail is green (BGR)
    colors = np.full_like(image, (180, 0, 180))
    colors[target == 0] = (35, 35, 35)
    colors[target == 1] = (0, 220, 0)
    comparison = cv2.hconcat([image, colors, overlay_mask(image, target, "Ground truth")])
    write_image(args.out, comparison)
    ids, counts = np.unique(target, return_counts=True)
    print("Sample:", names[args.index], "shape:", image.shape)
    print("Counts (0 other, 1 trail, 255 ignore):", dict(zip(ids.tolist(), counts.tolist())))
    print("Saved:", args.out)
    if args.show:
        cv2.namedWindow("Input | Target | Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Input | Target | Overlay", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
