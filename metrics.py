import numpy as np

# accept predictions and labels
def confusion_matrix(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")
    # unknown annotations contribute to neither correct predictions nor errors
    valid = target != 255
    actual = target[valid].astype(np.int64)
    predicted = prediction[valid].astype(np.int64)
    if np.any((actual < 0) | (actual > 1) | (predicted < 0) | (predicted > 1)):
        raise ValueError("Expected binary category IDs")
    # rows = truth, columns = prediction: [[TN, FP], [FN, TP]]
    # encode (truth, prediction) as 0, 1, 2, or 3, then count each pair
    return np.bincount(2 * actual + predicted, minlength=4).reshape(2, 2)


def scores(matrix):
    tn, fp, fn, tp = np.asarray(matrix, dtype=np.float64).ravel()

    def ratio(a, b):
        if b == 0:
            return None
        return float(a / b)

    # IoU is intersection / union, both false positives and false negatives is in there
    trail_iou = ratio(tp, tp + fp + fn)
    other_iou = ratio(tn, tn + fp + fn)
    # average only defined class scores, rather than replacing missing scores with zero
    present = [value for value in (trail_iou, other_iou) if value is not None]
    mean_iou = None
    if present:
        mean_iou = float(np.mean(present))
    return {
        "trail_iou": trail_iou,
        "other_iou": other_iou,
        "mean_iou": mean_iou,
        "trail_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "trail_precision": ratio(tp, tp + fp),
        "trail_recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tn + fp + fn + tp),
        "valid_pixels": int(tn + fp + fn + tp),
        "confusion_matrix": np.asarray(matrix, dtype=np.int64).tolist(),
    }
