# initialize dataset
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DATA = Path(__file__).resolve().parent / "data/freiburg_forest_annotated"
IGNORE = 255
CLASS_NAMES = ["other", "trail"]
# ImageNet channel statistics, shaped [3, 1, 1] to broadcast over image height/width.
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
# BGR values from the downloaded dataset README (which lists RGB).
OTHER_COLORS = [(0, 255, 0), (51, 102, 102), (0, 60, 0),
                (255, 120, 0), (0, 0, 0)]


def mask_to_target(mask):
    # Match all three color channels; unknown colors stay ignored rather than
    # silently becoming non-trail training examples.
    target = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    target[(mask == (170, 170, 170)).all(axis=2)] = 1
    for color in OTHER_COLORS:
        target[(mask == color).all(axis=2)] = 0
    return target


def list_images(root, split="train"):
    paths = sorted((Path(root) / split / "rgb").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images in {Path(root) / split / 'rgb'}")
    return [path.name for path in paths]


def read_pair(root, split, filename):
    folder = Path(root) / split
    stem = Path(filename).stem.removesuffix("_Clipped")
    image_path = folder / "rgb" / filename
    # The archive uses two annotation suffixes; require exactly one matching mask.
    candidates = [folder / "GT_color" / f"{stem}{suffix}.png"
                  for suffix in ("_mask", "_Clipped")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Expected one matching annotation for {filename}; found {existing}")
    mask_path = existing[0]
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))
    if image is None or mask is None:
        raise FileNotFoundError(f"Cannot read pair: {image_path}, {mask_path}")
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {filename}")
    return image, mask_to_target(mask)


def make_split(filenames, seed=42, val_fraction=0.2):

    groups = sorted({name.split("-")[0] for name in filenames})
    if len(groups) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two groups and 0 < val_fraction < 1")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
    val_groups = set(groups[:count])
    train = [name for name in filenames if name.split("-")[0] not in val_groups]
    val = [name for name in filenames if name.split("-")[0] in val_groups]
    return {"train": train, "val": val, "seed": seed,
            "method": "filename-prefix grouping (recording independence unverified)"}


def image_to_tensor(image, size):
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert HWC bytes to CHW floats in [0, 1] before channel normalization.
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - MEAN) / STD


class ForestDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA, split="train", filenames=None,
                 size=(256, 448), augment=False):
        self.root, self.split = Path(root), split
        # explicit filenames let training and validation share a source folder
        # while keeping their saved image lists separate
        if filenames is None:
            self.filenames = list_images(root, split)
        else:
            self.filenames = filenames
        self.size, self.augment = tuple(size), augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image, target = read_pair(self.root, self.split, self.filenames[index])
        if self.augment:
            # apply the same geometric change to the image and its labels
            if torch.rand(()).item() < 0.5:
                image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            brightness = 0.8 + 0.4 * torch.rand(()).item()
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        tensor = image_to_tensor(image, self.size)
        target = cv2.resize(target, (self.size[1], self.size[0]),
                            interpolation=cv2.INTER_NEAREST)
        # Cross-entropy expects an image [3, H, W] and integer target [H, W]
        return tensor, torch.from_numpy(target.astype(np.int64))
