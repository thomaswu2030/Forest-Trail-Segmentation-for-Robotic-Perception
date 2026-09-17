# Use this file to see what a sample training picture would look like
from pathlib import Path
import cv2
import numpy

# Locate the training data relative to this script.
data_dir = (
    Path(__file__).resolve().parent
    / ".."
    / "data"
    / "freiburg_forest_annotated"
    / "train"
)

# Select the first RGB image.
image_path = sorted((data_dir / "rgb").glob("*.jpg"))[0]

# Match its filename to the corresponding annotation.
# Check the image and mask file to see how they match
sample_name = image_path.stem.removesuffix("_Clipped")
mask_path = data_dir / "GT_color" / f"{sample_name}_mask.png"

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path))

if image is None or mask is None:
    raise FileNotFoundError(
        f"Could not load the image pair:\n{image_path}\n{mask_path}"
    )

print("Image:", image_path.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

matches = mask == [170, 170, 170]
trail_mask = matches.all(axis=2)

print(trail_mask.shape)
print(trail_mask.dtype)

trail_image = []
pixel_row = []
for row in trail_mask:
    pixel_row = []
    for pixel in row:
        if pixel:
            pixel_row.append([0, 0, 255])
        else:
            pixel_row.append([255, 255, 255])
    trail_image.append(pixel_row)

trail_image = numpy.array(trail_image, dtype=numpy.uint8)

print(trail_image.shape)
print(trail_image.dtype)

# Combine them horizontally: photograph on the left, mask on the right.
comparison = cv2.hconcat([image, mask, trail_image])

cv2.namedWindow("Image | Ground truth", cv2.WINDOW_NORMAL)
cv2.imshow("Image | Ground truth", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
