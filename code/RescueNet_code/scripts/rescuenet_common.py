"""Shared, test-sealed utilities for RescueNet classification experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["superficial_damage", "medium_damage", "major_damage"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_rows(csv_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split: {csv_path}")
    result: list[dict[str, Any]] = []
    for row in rows:
        label = int(row["label_id"])
        if label not in range(len(CLASS_NAMES)) or row["label"] != CLASS_NAMES[label]:
            raise ValueError(f"Invalid class mapping in {csv_path}: {row}")
        path = (dataset_root / row["relative_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(
            {
                **row,
                "label_id": label,
                "image_path": str(path),
                "training_path": str(
                    (dataset_root / "derived" / "draft256_jpeg95" / row["image_id"]).resolve()
                    if (dataset_root / "derived" / "draft256_jpeg95" / row["image_id"]).is_file()
                    else path
                ),
                "split_csv": str(csv_path.resolve()),
            }
        )
    if len({row["image_id"] for row in result}) != len(result):
        raise ValueError(f"Duplicate image IDs in {csv_path}")
    return result


class RescueNetDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        with Image.open(row["training_path"]) as image:
            # RescueNet JPEGs are 4000x3000. Ask the JPEG decoder for its
            # native downsampled draft before the 224px augmentation pipeline;
            # this avoids repeatedly materializing 12 MP arrays on CPU while
            # preserving substantially more pixels than the network input.
            image.draft("RGB", (256, 256))
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label_id"]), index


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
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    precision, recall, per_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    two_level_count = int(np.sum(np.abs(labels_array - predictions_array) == 2))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(labels_array - predictions_array))),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "mild_severe_error_count": two_level_count,
        "mild_severe_error_rate": float(two_level_count / max(len(labels), 1)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASS_NAMES))
        },
    }


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label_id"]) for row in rows)
    return {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))}
