"""Build frozen train-only grouped 3-fold CE OOF confusions for AIDER."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

import train_ce


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class IndexedDataset(Dataset):
    def __init__(self, rows, transform):
        self.base = train_ce.SceneDataset(rows, transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


def metrics(labels, predictions, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(class_names)))).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs != 50 or args.batch_size != 32 or args.learning_rate != 3e-4 or args.weight_decay != 1e-4:
        parser.error("Frozen AIDER protocol requires 50 epochs, batch 32, AdamW lr=3e-4, wd=1e-4")
    if args.seed != 42:
        parser.error("This OOF source is frozen to seed42")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    output = args.output_dir
    if (output / "oof_audit.json").exists():
        raise FileExistsError(f"Completed OOF already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    clock = time.perf_counter()
    rows_by_split, class_names = train_ce.load_rows(args.split_file, None)
    rows = rows_by_split["train"]
    if any(row["split"] != "train" for row in rows):
        raise ValueError("OOF input contains a non-train row")
    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([row["group_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    folds = np.full(len(rows), -1, dtype=np.int64)
    splits = list(splitter.split(np.zeros(len(rows)), labels, groups))
    for fold, (_, holdout_indices) in enumerate(splits):
        folds[holdout_indices] = fold
    if np.any(folds < 0):
        raise RuntimeError("Incomplete OOF fold assignment")
    for fold in range(3):
        fit_groups = set(groups[folds != fold])
        holdout_groups = set(groups[folds == fold])
        if fit_groups & holdout_groups:
            raise RuntimeError("OOF group leakage")
    with (output / "fold_assignments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "group_id", "class_name", "label", "fold"])
        writer.writeheader()
        for row, fold in zip(rows, folds):
            writer.writerow({"relative_path": row["relative_path"], "group_id": row["group_id"], "class_name": row["class_name"], "label": row["label"], "fold": int(fold) + 1})
    train_transform, val_transform = train_ce.build_transforms(224)
    device = torch.device(args.device)
    predictions = []
    fold_metrics = []
    for fold in range(3):
        fold_clock = time.perf_counter()
        train_ce.set_seed(args.seed + fold + 1)
        fit_rows = [row for i, row in enumerate(rows) if folds[i] != fold]
        holdout_rows = [row for i, row in enumerate(rows) if folds[i] == fold]
        fit_loader = DataLoader(train_ce.SceneDataset(fit_rows, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        holdout_loader = DataLoader(IndexedDataset(holdout_rows, val_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=len(class_names)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            count = 0
            for images, batch_labels in fit_loader:
                images, batch_labels = images.to(device), batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), batch_labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_labels)
                count += len(batch_labels)
            scheduler.step()
            print(f"fold={fold+1}/3 epoch={epoch:03d}/50 loss={loss_sum/count:.6f}", flush=True)
        model.eval()
        fold_labels, fold_predictions = [], []
        with torch.inference_mode():
            for images, batch_labels, indices in holdout_loader:
                probs = F.softmax(model(images.to(device)), dim=1).cpu()
                preds = probs.argmax(1)
                for position, index in enumerate(indices.tolist()):
                    row = holdout_rows[index]
                    record = {
                        "relative_path": row["relative_path"], "group_id": row["group_id"],
                        "class_name": row["class_name"], "label": int(batch_labels[position]),
                        "prediction": int(preds[position]), "predicted_class": class_names[int(preds[position])],
                        "confidence": float(probs[position, int(preds[position])]), "fold": fold + 1,
                    }
                    predictions.append(record)
                    fold_labels.append(record["label"])
                    fold_predictions.append(record["prediction"])
        fold_metrics.append({
            "fold": fold + 1, "seed": args.seed + fold + 1, "fit_samples": len(fit_rows),
            "holdout_samples": len(holdout_rows), "fit_groups": len(set(row["group_id"] for row in fit_rows)),
            "holdout_groups": len(set(row["group_id"] for row in holdout_rows)),
            "duration_seconds": time.perf_counter() - fold_clock,
            "metrics": metrics(fold_labels, fold_predictions, class_names),
        })
        del model
    if len(predictions) != len(rows) or len({item["relative_path"] for item in predictions}) != len(rows):
        raise RuntimeError("OOF coverage incomplete or duplicated")
    predictions.sort(key=lambda item: item["relative_path"])
    with (output / "oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    y_true = [item["label"] for item in predictions]
    y_pred = [item["prediction"] for item in predictions]
    overall = metrics(y_true, y_pred, class_names)
    matrix = np.asarray(overall["confusion_matrix"], dtype=np.int64)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    for name, value in (("confusion_raw.csv", matrix), ("confusion_row_normalized.csv", normalized)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["true/predicted", *class_names])
            for class_name, row in zip(class_names, value): writer.writerow([class_name, *row.tolist()])
    pairs = []
    for index, class_name in enumerate(class_names):
        rates = normalized[index].copy(); rates[index] = -1
        opponent = int(np.argmax(rates))
        pairs.append({
            "target_label": index, "target_class": class_name, "opponent_label": opponent,
            "opponent_class": class_names[opponent], "normalized_confusion_rate": float(normalized[index, opponent]),
            "raw_confusion_count": int(matrix[index, opponent]), "fallback_used": False,
            "source": "train_only_grouped_3fold_oof",
        })
    write_json(output / "confusion_pairs.json", pairs)
    audit = {
        "schema_version": "1.0.0", "completion_signal": "AIDER_TOP1_OOF_COMPLETE",
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - clock, "protocol": "train_only_grouped_3fold_oof",
        "split_file": str(args.split_file.resolve()), "split_sha256": sha256(args.split_file),
        "fold_assignments_sha256": sha256(output / "fold_assignments.csv"), "fold_count": 3,
        "sample_count": len(rows), "group_count": len(set(groups)), "fold_metrics": fold_metrics,
        "overall_metrics": overall, "validation_used": False, "test_used": False,
        "test_images_read": 0, "remoteclip_used": False, "epochs_per_fold": 50, "seed": 42,
    }
    write_json(output / "oof_audit.json", audit)
    print("AIDER_TOP1_OOF_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
