from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from dataset import ForestDataset, image_to_tensor, make_split, mask_to_target
from metrics import confusion_matrix, scores
from model import build_model, load_checkpoint


# Small synthetic examples exercise the pipeline without downloading data or pretrained weights
class PipelineTests(unittest.TestCase):
    # verify all known colors, plus unknown white and a near-match to the trail color.
    def test_palette_and_unknowns(self):
        # all documented BGR colors
        mask = np.array([[(170, 170, 170), (0, 255, 0), (51, 102, 102),
                          (0, 60, 0), (255, 120, 0), (0, 0, 0),
                          (255, 255, 255), (170, 170, 169)]], dtype=np.uint8)
        np.testing.assert_array_equal(mask_to_target(mask), [[1, 0, 0, 0, 0, 0, 255, 255]])

    # TN=1, FP=1, FN=1, TP=2, so trail IoU is 2/4.
    def test_ignore_pixels_do_not_change_metrics(self):
        truth = np.array([[0, 0, 1], [1, 1, 255]])
        pred = np.array([[0, 1, 0], [1, 1, 0]])
        matrix = confusion_matrix(pred, truth)
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 2]])
        self.assertEqual(scores(matrix)["trail_iou"], 0.5)
        pred[1, 2] = 1
        np.testing.assert_array_equal(confusion_matrix(pred, truth), matrix)
        self.assertIsNone(scores(np.zeros((2, 2)))["trail_iou"])

    # generate ten filename groups with three frames each to test grouped splitting
    def test_split_keeps_prefix_groups_together(self):
        names = [f"b{group}-{frame}_Clipped.jpg" for group in range(10) for frame in range(3)]
        split = make_split(names)
        self.assertEqual(split, make_split(names))
        train_groups = {name.split("-")[0] for name in split["train"]}
        val_groups = {name.split("-")[0] for name in split["val"]}
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(set(split["train"] + split["val"]), set(names))

    # a disposable miniature dataset to exercise actual image reading and pairing.
    def test_dataset_pairing_normalization_and_label_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train/rgb").mkdir(parents=True)
            (root / "train/GT_color").mkdir(parents=True)
            # this is a red BGR image with trail on the left half and other on the right.
            image = np.full((4, 6, 3), (0, 0, 255), dtype=np.uint8)
            annotation = np.zeros_like(image)
            annotation[:, :3] = (170, 170, 170)
            cv2.imwrite(str(root / "train/rgb/b1-1_Clipped.jpg"), image)
            cv2.imwrite(str(root / "train/GT_color/b1-1_mask.png"), annotation)
            # double both dimensions
            inputs, labels = ForestDataset(root, size=(8, 12))[0]
            self.assertEqual(tuple(inputs.shape), (3, 8, 12))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertTrue(torch.all(labels[:, :6] == 1))
            self.assertTrue(torch.all(labels[:, 6:] == 0))
            # check RGB channel order and normalization using the original array,
            # avoiding JPEG compression differences in this numerical assertion.
            red = image_to_tensor(image, (4, 6))
            self.assertAlmostEqual(red[0, 0, 0].item(), (1 - 0.485) / 0.229, places=5)
            self.assertAlmostEqual(red[2, 0, 0].item(), (0 - 0.406) / 0.225, places=5)
            # Both supported annotation suffixes must resolve to identical labels.
            (root / "train/GT_color/b1-1_mask.png").rename(root / "train/GT_color/b1-1_Clipped.png")
            _, alternate = ForestDataset(root, size=(8, 12))[0]
            torch.testing.assert_close(labels, alternate)

    # verify gradients reach both classifier branches and saved weights reproduce outputs
    def test_backward_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        model = build_model(pretrained=False).eval()
        inputs = torch.randn(1, 3, 32, 48)
        targets = torch.zeros(1, 32, 48, dtype=torch.long)
        targets[:, :, 24:] = 1
        targets[:, :2] = 255
        logits = model(inputs)["out"]
        self.assertEqual(tuple(logits.shape), (1, 2, 32, 48))
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=255)
        loss.backward()
        # both prediction branches must receive finite gradients from cross-entropy
        for branch in (model.classifier.low_classifier, model.classifier.high_classifier):
            self.assertIsNotNone(branch.weight.grad)
            self.assertTrue(torch.isfinite(branch.weight.grad).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            # save the minimal format accepted by load_checkpoint, then reload on CPU
            torch.save({"format_version": 1, "classes": ["other", "trail"],
                        "model": model.state_dict()}, path)
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
            with torch.inference_mode():
                # restore
                torch.testing.assert_close(logits.detach(), loaded(inputs)["out"])


if __name__ == "__main__":
    unittest.main()
