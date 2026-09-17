import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, DEFAULT_DATA, IGNORE, ForestDataset, list_images, make_split
from metrics import confusion_matrix, scores
from model import build_model, choose_device


def run_epoch(model, loader, device, optimizer=None):
    # share the loop, validation changes model mode and disables gradient updates.
    training = optimizer is not None
    model.train(training)
    if training:
        # small batches make BatchNorm running-statistic estimates noisy
        # Keep the pretrained statistics
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    total_loss, pixels = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            valid_count = int((targets != IGNORE).sum().item())
            if valid_count == 0:
                continue  # avoid NaN from averaging over an entirely ignored batch
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            # cross-entropy expects raw class scores and integer pixel labels
            # It applies the required log-softmax internally.
            loss = nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss: inspect data and learning rate")
            if training:
                loss.backward()  # Compute gradients of loss with respect to weights
                optimizer.step()  # Use those gradients to adjust the weights
            total_loss += loss.item() * valid_count
            pixels += valid_count
            matrix += confusion_matrix(logits.detach().argmax(1).cpu().numpy(),
                                       targets.cpu().numpy())
    if not pixels:
        raise ValueError("No labeled pixels found in this epoch")
    return {"loss": total_loss / pixels, **scores(matrix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0, help="0 is simplest on Windows")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="Smoke test only; 0 = all")
    parser.add_argument("--limit-val", type=int, default=0, help="Smoke test only; 0 = all")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.height, args.width) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, dimensions and learning rate must be positive")
    if min(args.height, args.width) < 32 or min(args.workers, args.limit_train, args.limit_val) < 0:
        parser.error("Dimensions must be >=32; workers and limits must be nonnegative")
    if (args.out / "config.json").exists():
        parser.error("This run already exists. Choose a new --out directory.")

    # seed sampling and initialization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    split = make_split(list_images(args.data), args.seed)
    if args.limit_train:
        split["train"] = split["train"][:args.limit_train]
    if args.limit_val:
        split["val"] = split["val"][:args.limit_val]
    size = (args.height, args.width)
    train_data = ForestDataset(args.data, filenames=split["train"], size=size,
                               augment=not args.no_augment)
    val_data = ForestDataset(args.data, filenames=split["val"], size=size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_data, args.batch_size, num_workers=args.workers)
    model = build_model().to(device)
    # Fine-tune pretrained features more slowly than the new classifiers
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=0.0001)

    args.out.mkdir(parents=True, exist_ok=True)
    # JSON cannot encode Path objects, so convert just those settings to text
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        config[key] = value
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    config.update(torch_version=str(torch.__version__), device=str(device),
                  gpu=gpu_name,
                  classes=CLASS_NAMES, ignore_index=IGNORE, batchnorm="frozen running statistics")
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Device: {device}; train: {len(train_data)}; validation: {len(val_data)}", flush=True)
    print("Validation is resized to training dimensions. Test data remains separate.", flush=True)
    best_iou = -1.0
    start = time.perf_counter()
    fields = ["epoch", "seconds", "train_loss", "val_loss", "val_trail_iou", "val_mean_iou"]
    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_result = run_epoch(model, train_loader, device, optimizer)
            val_result = run_epoch(model, val_loader, device)
            value = val_result["trail_iou"]
            if value is None:
                raise ValueError("Validation has no trail union; choose a representative split")
            # choose the best model using validation trail IoU, leaving test data unused
            improved = value > best_iou
            if improved:
                best_iou = value
            # save CPU weights and inference metadata. Optimizer state is not stored
            checkpoint = {
                "format_version": 1, "classes": CLASS_NAMES, "ignore_index": IGNORE,
                "model": {key: val.detach().cpu() for key, val in model.state_dict().items()},
                "epoch": epoch, "size": list(size), "config": config, "split": split,
                "val_metrics": val_result,
            }
            torch.save(checkpoint, args.out / "last.pt")
            if improved:
                torch.save(checkpoint, args.out / "best.pt")
            row = dict(epoch=epoch, seconds=round(time.perf_counter() - epoch_start, 2),
                       train_loss=train_result["loss"], val_loss=val_result["loss"],
                       val_trail_iou=value, val_mean_iou=val_result["mean_iou"])
            writer.writerow(row)
            handle.flush()
            best_marker = ""
            if improved:
                best_marker = " [best]"
            print(f"Epoch {epoch}/{args.epochs}: train loss {row['train_loss']:.4f}, "
                  f"val loss {row['val_loss']:.4f}, trail IoU {value:.3f} "
                  f"({row['seconds']}s){best_marker}", flush=True)
    print(f"Finished in {time.perf_counter() - start:.1f}s. Best checkpoint: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
