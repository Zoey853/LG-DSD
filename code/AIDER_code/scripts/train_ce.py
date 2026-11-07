"""Train and validate a MobileNetV3 CE baseline on the sealed AIDER split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
AIDER_DIR = SCRIPT_DIR.parent
DEFAULT_SPLIT = AIDER_DIR / "splits" / "aider_seed42_grouped_70_10_20.csv"
DEFAULT_OUTPUT = AIDER_DIR / "outputs" / "aider_E1_ce"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SceneDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recipe", choices=("standard", "class_weighted"), required=True)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(split_file: Path, max_samples_per_class: int | None) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    with split_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split file: {split_file}")
    class_map = {int(row["label"]): row["class_name"] for row in rows}
    class_names = [class_map[index] for index in sorted(class_map)]
    selected: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        split = row["split"]
        if split not in selected:
            continue
        key = (split, int(row["label"]))
        if max_samples_per_class is not None and counts[key] >= max_samples_per_class:
            continue
        selected[split].append(row)
        counts[key] += 1
    if not selected["train"] or not selected["val"]:
        raise ValueError("Both train and val rows are required; test rows are intentionally ignored")
    return selected, class_names


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def normalized_inverse_frequency_weights(rows: list[dict[str, str]], class_count: int) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    weights = torch.tensor([len(rows) / (class_count * counts[index]) for index in range(class_count)])
    return weights / weights.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * labels.size(0)
        sample_count += labels.size(0)
    return loss_sum / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    class_weights: torch.Tensor | None,
) -> dict[str, object]:
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        loss_sum += float(loss) * labels.size(0)
        sample_count += labels.size(0)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels_all,
        predictions_all,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return {
        "loss": loss_sum / sample_count,
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(
            labels_all, predictions_all, labels=list(range(len(class_names)))
        ).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
    start_clock = time.perf_counter()
    set_seed(args.seed)
    rows, class_names = load_rows(args.split_file, args.max_samples_per_class)
    train_transform, val_transform = build_transforms(args.image_size)
    train_loader = DataLoader(
        SceneDataset(rows["train"], train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        SceneDataset(rows["val"], val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    device = torch.device(args.device)
    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=len(class_names)).to(device)
    weights_cpu = normalized_inverse_frequency_weights(rows["train"], len(class_names))
    class_weights = weights_cpu.to(device) if args.recipe == "class_weighted" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = args.output_dir / f"{args.model_name}_{args.recipe}_pretrained{int(args.pretrained)}_seed{args.seed}"
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"Completed run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        validation = evaluate(model, val_loader, device, class_names, class_weights)
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "duration_seconds": time.perf_counter() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation": validation,
        }
        history.append(epoch_record)
        macro_f1 = float(validation["macro_f1"])
        accuracy = float(validation["accuracy"])
        is_better = macro_f1 > best_macro_f1 or (
            np.isclose(macro_f1, best_macro_f1) and accuracy > best_accuracy
        )
        if is_better:
            best_epoch = epoch
            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "validation": validation,
                },
                run_dir / "best.pt",
            )
        progress = {
            "completion_signal": "RUNNING",
            "epoch": epoch,
            "epochs": args.epochs,
            "recipe": args.recipe,
            "latest_validation_accuracy": accuracy,
            "latest_validation_macro_f1": macro_f1,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "best_validation_macro_f1": best_macro_f1,
            "elapsed_seconds": time.perf_counter() - start_clock,
            "test_used": False,
        }
        write_json(run_dir / "progress.json", progress)
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.5f} "
            f"val_accuracy={accuracy:.5f} val_macro_f1={macro_f1:.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    completed_at = datetime.now(UTC)
    result = {
        "schema_version": "1.0.0",
        "experiment": "AIDER_E1_CE_recipe_screen",
        "completion_signal": "AIDER_E1_CE_COMPLETE",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": time.perf_counter() - start_clock,
        "recipe": args.recipe,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "device": str(device),
        "split_file": str(args.split_file.resolve()),
        "sample_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "class_names": class_names,
        "class_weights": weights_cpu.tolist() if args.recipe == "class_weighted" else None,
        "selection_metric": "validation_macro_f1",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro_f1,
        "final_validation_accuracy": history[-1]["validation"]["accuracy"],
        "final_validation_macro_f1": history[-1]["validation"]["macro_f1"],
        "history": history,
        "test_used": False,
        "test_policy": "sealed_until_method_and_recipe_are_frozen",
    }
    write_json(run_dir / "result.json", result)
    write_json(
        run_dir / "progress.json",
        {
            "completion_signal": "AIDER_E1_CE_COMPLETE",
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "best_validation_macro_f1": best_macro_f1,
            "duration_seconds": result["duration_seconds"],
            "test_used": False,
        },
    )
    print("AIDER_E1_CE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
