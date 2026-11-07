"""Generate leakage-safe out-of-fold confusion statistics from the AID train split."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from train_student import (
    AIDStudentDataset,
    DEFAULT_SPLIT_FILE,
    build_transforms,
    load_split_rows,
    set_seed,
    train_one_epoch,
)


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CODE_DIR / "outputs" / "aid_train_oof_confusions_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    return args


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[list[int], list[int], list[list[float]]]:
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[list[float]] = []
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = F.softmax(logits, dim=1)
        labels_all.extend(labels.tolist())
        predictions_all.extend(probabilities.argmax(dim=1).cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
    return labels_all, predictions_all, probabilities_all


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)


def build_confusable_classes(
    matrix: np.ndarray, class_names: list[str], top_k: int
) -> list[dict[str, object]]:
    row_rates = normalized_rows(matrix)
    class_totals = matrix.sum(axis=1)
    records: list[dict[str, object]] = []
    for label, class_name in enumerate(class_names):
        candidates = []
        for other_label, other_name in enumerate(class_names):
            if other_label == label:
                continue
            forward_count = int(matrix[label, other_label])
            reverse_count = int(matrix[other_label, label])
            forward_rate = float(row_rates[label, other_label])
            reverse_rate = float(row_rates[other_label, label])
            candidates.append(
                {
                    "label": other_label,
                    "class_name": other_name,
                    "bidirectional_count": forward_count + reverse_count,
                    "forward_count": forward_count,
                    "reverse_count": reverse_count,
                    "forward_rate": forward_rate,
                    "reverse_rate": reverse_rate,
                    "bidirectional_rate": forward_rate + reverse_rate,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["bidirectional_rate"]),
                int(item["bidirectional_count"]),
                float(item["forward_rate"]),
            ),
            reverse=True,
        )
        records.append(
            {
                "label": label,
                "class_name": class_name,
                "num_oof_samples": int(class_totals[label]),
                "top_confusable": candidates[:top_k],
            }
        )
    return records


def write_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_class/predicted_class", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    rows_by_split, class_names = load_split_rows(
        args.split_file, args.max_samples_per_class
    )
    train_rows = rows_by_split["train"]
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    if len(train_rows) != len(set(str(row["image_path"]) for row in train_rows)):
        raise ValueError("Duplicate image paths exist in the training split.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_transform, evaluation_transform = build_transforms(args.image_size)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    oof_records: list[dict[str, object] | None] = [None] * len(train_rows)
    fold_metrics: list[dict[str, object]] = []

    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        fit_rows = [train_rows[index] for index in fit_indices]
        holdout_rows = [train_rows[index] for index in holdout_indices]
        fit_loader = DataLoader(
            AIDStudentDataset(fit_rows, train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        holdout_loader = DataLoader(
            AIDStudentDataset(holdout_rows, evaluation_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=len(class_names),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        print(
            f"Fold {fold}/{args.folds}: fit={len(fit_rows)}, "
            f"holdout={len(holdout_rows)}, seed={fold_seed}"
        )
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                model,
                fit_loader,
                optimizer,
                device,
                "ce",
                0.0,
                1.0,
                1.0,
                "none",
                kd_weighting="none",
                kd_weight_floor=0.05,
                kd_incorrect_weight=0.0,
                kd_agreement_weighting="none",
                kd_agreement_floor=0.1,
            )
            scheduler.step()
            print(f"Fold {fold:02d} epoch {epoch:03d}: loss={metrics['loss']:.4f}")

        fold_labels, fold_predictions, fold_probabilities = predict(
            model, holdout_loader, device
        )
        for local_index, global_index in enumerate(holdout_indices):
            row = train_rows[int(global_index)]
            oof_records[int(global_index)] = {
                "image_path": str(row["image_path"]),
                "class_name": str(row["class_name"]),
                "label": int(fold_labels[local_index]),
                "prediction": int(fold_predictions[local_index]),
                "predicted_class": class_names[int(fold_predictions[local_index])],
                "fold": fold,
                "confidence": max(fold_probabilities[local_index]),
            }
        fold_metrics.append(
            {
                "fold": fold,
                "fit_samples": len(fit_rows),
                "holdout_samples": len(holdout_rows),
                "seed": fold_seed,
                "accuracy": accuracy_score(fold_labels, fold_predictions),
                "macro_f1": f1_score(
                    fold_labels, fold_predictions, average="macro", zero_division=0
                ),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(record is None for record in oof_records):
        raise RuntimeError("OOF prediction coverage is incomplete.")
    completed_records = [record for record in oof_records if record is not None]
    oof_labels = [int(record["label"]) for record in completed_records]
    oof_predictions = [int(record["prediction"]) for record in completed_records]
    matrix = confusion_matrix(
        oof_labels, oof_predictions, labels=list(range(len(class_names)))
    )
    row_matrix = normalized_rows(matrix)
    confusable_classes = build_confusable_classes(matrix, class_names, args.top_k)

    with (args.output_dir / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(completed_records[0].keys()))
        writer.writeheader()
        writer.writerows(completed_records)
    write_matrix(args.output_dir / "confusion_counts.csv", matrix, class_names)
    write_matrix(args.output_dir / "confusion_row_normalized.csv", row_matrix, class_names)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "stratified_training_split_out_of_fold_predictions",
        "split_file": str(args.split_file.resolve()),
        "source_split": "train_only",
        "validation_split_used": False,
        "test_split_used": False,
        "args": vars(args),
        "class_names": class_names,
        "num_train_samples": len(train_rows),
        "num_oof_predictions": len(completed_records),
        "unique_oof_image_paths": len(
            set(str(record["image_path"]) for record in completed_records)
        ),
        "oof_accuracy": accuracy_score(oof_labels, oof_predictions),
        "oof_macro_f1": f1_score(
            oof_labels, oof_predictions, average="macro", zero_division=0
        ),
        "fold_metrics": fold_metrics,
        "confusable_classes": confusable_classes,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
    print(f"OOF accuracy: {summary['oof_accuracy']:.4f}")
    print(f"OOF macro-F1: {summary['oof_macro_f1']:.4f}")
    print(f"Saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
