# Forest Trail Segmentation

A forest trail segmentation project using PyTorch, torchvision, and OpenCV.
It adapts a pretrained LR-ASPP model with a MobileNetV3 backbone to classify
each pixel as trail or other, with tools for training, evaluation, and
image or video prediction.

## Set up

Run the commands below from the project root, the folder containing `train.py`.

Create a virtual environment if you do not already have one:

```powershell
python -m venv .venv
```

Install compatible PyTorch and torchvision packages for your CPU or CUDA
environment, then install the remaining requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` contains NumPy and OpenCV. PyTorch and torchvision are
installed separately. My GPU runs used PyTorch `2.11.0+cu128` and
torchvision `0.26.0+cu128`. Training downloads pretrained weights on first use and
loading a saved project checkpoint does not download weights.

### Dataset

The scripts expect the Freiburg Forest annotated dataset in this layout:

```text
data/
└── freiburg_forest_annotated/
    ├── train/
    │   ├── rgb/
    │   └── GT_color/
    └── test/
        ├── rgb/
        └── GT_color/
```

The included downloader combines the archive parts in `Tools/data`. Extract
the combined archive into the project-level `data` directory:

```powershell
.\.venv\Scripts\python.exe Tools/data_downloader.py
New-Item -ItemType Directory -Force data
tar -xzf Tools/data/freiburg_forest_annotated.tar.gz -C data
```

Confirm the extracted folders match the layout above.

### Preview annotations

View an image alongside its decoded ground-truth labels before training:

```powershell
.\.venv\Scripts\python.exe preview.py --out outputs/preview.jpg
```

This displays human annotations. Add `--show` to open
a preview window.

### Train

Run training in PowerShell:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 10 --batch-size 4 --height 256 --width 448 --out runs/baseline
```

Training creates a validation split from the training images and selects the
best checkpoint using validation trail IoU. The checkpoint is saved as
`runs/baseline/best.pt`. The official test split is reserved for evaluation.

### Evaluate

Evaluate on the saved validation split:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/baseline/best.pt --split val --out outputs/validation
```

After choosing a model using validation results, evaluate the official test set:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/baseline/best.pt --split test --out outputs/test
```

### Predict an image or video

Replace the input paths below with your own files:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint runs/baseline/best.pt --input photo.jpg --output outputs/prediction.jpg
.\.venv\Scripts\python.exe predict.py --checkpoint runs/baseline/best.pt --input video.mp4 --output outputs/prediction.mp4
```

Image prediction saves an overlay and a binary mask.

### Useful options

| Option | Available in | Purpose |
| --- | --- | --- |
| `--device auto` | Train, evaluate, predict | Use CUDA when available, otherwise CPU. This is the default. |
| `--device cpu` / `--device cuda` | Train, evaluate, predict | Select a device explicitly. |
| `--data PATH` | Preview, train, evaluate | Set the dataset root. |
| `--height H --width W` | Train | Set the model input size; defaults to `256 × 448`. |
| `--seed 42` | Train | Set the random seed; defaults to `42`. |
| `--show` | Preview, predict | Open a display window; press `Q` to exit video playback. |
| `--max-frames N` | Predict | Limit video processing; `0` processes the entire video. |
| `--help` | All four scripts | List the script's available arguments. |

## Label conventions

| Value | Meaning |
| --- | --- |
| `0` | Other: known non-trail annotation colors. |
| `1` | Trail: annotation color `(170, 170, 170)`. |
| `255` | Unknown annotation color, ignored during loss and metric calculation. |

## Recorded results

The reference `256 × 448` model achieved the following results on the 136-image
official test split, as recorded in [RESULTS.md](RESULTS.md):

| Metric | Result |
| --- | ---: |
| Trail IoU | 86.88% |
| Trail Dice | 92.98% |
| Trail precision | 93.08% |
| Trail recall | 92.89% |
