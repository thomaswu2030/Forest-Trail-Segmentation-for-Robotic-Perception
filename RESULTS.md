# Reference run results

Measured on 2026-09-16 with the RTX 5070 Ti Laptop GPU, PyTorch 2.11.0+cu128,
and torchvision 0.26.0+cu128. These are actual local runs.

## Controlled resolution comparison

Both runs used seed 42, exactly the same 185 training / 45 validation images,
10 epochs, batch size 4, pretrained LR-ASPP/MobileNetV3, AdamW, paired flips,
and brightness augmentation. Each run selected its best epoch using validation
trail IoU at training resolution, and both selected epoch 10

The following evaluation scores use the **original annotation resolution**:

| Input height x width | Validation trail IoU | Trail precision | Trail recall | Prediction FPS |
|---|---:|---:|---:|---:|
| 128 x 224 | 85.32% | 94.91% | 89.41% | 183.7 |
| 256 x 448 | **88.29%** | 94.17% | **93.40%** | 125.8 |

For this seeded experiment, the larger input recovered more trail pixels and
improved IoU by about 3 percentage points, but the smaller input predicted faster.

## Final official test evaluation

The 256x448 checkpoint was chosen using validation IoU, then evaluated on all
136 official test images without further training or parameter changes:

| Metric | Result |
|---|---:|
| Trail IoU | **86.88%** |
| Trail Dice | 92.98% |
| Trail precision | 93.08% |
| Trail recall | 92.89% |
| Mean IoU (trail and other) | 92.79% |
| Pixel accuracy | 98.80% |

An all-other predictor achieves **91.48% pixel accuracy but 0% trail IoU** on this
test set. This illustrates why accuracy alone would be misleading.