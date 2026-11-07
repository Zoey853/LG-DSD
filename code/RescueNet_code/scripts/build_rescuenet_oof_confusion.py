"""Build preregistered grouped three-fold, train-only RescueNet OOF confusion."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rescuenet_common import (
    CLASS_NAMES,
    RescueNetDataset,
    build_transforms,
    classification_metrics,
    load_split_rows,
    set_seed,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.folds != 3:
        parser.error("The frozen protocol requires exactly three folds")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def grouped_fold_assignment(rows, folds: int, seed: int):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["group_id"])].append(index)
    group_stats = {}
    for group_id, indices in groups.items():
        counter = Counter({"total": len(indices), "groups": 1})
        counter.update(int(rows[index]["label_id"]) for index in indices)
        group_stats[group_id] = counter
    totals = Counter({"total": len(rows), "groups": len(groups)})
    totals.update(int(row["label_id"]) for row in rows)
    targets = {fold: Counter({key: value / folds for key, value in totals.items()}) for fold in range(folds)}

    def score(assignment):
        current = {fold: Counter() for fold in range(folds)}
        for group_id, fold in assignment.items():
            current[fold].update(group_stats[group_id])
        value = 0.0
        for fold in range(folds):
            for key, target in targets[fold].items():
                weight = 0.05 if key == "groups" else 1.0
                value += weight * ((current[fold][key] - target) / max(float(target), 1.0)) ** 2
        return value

    rng = random.Random(seed)
    group_ids = list(groups)
    best, best_score = None, float("inf")
    for _ in range(5000):
        rng.shuffle(group_ids)
        assignment = {}
        for group_id in sorted(group_ids, key=lambda item: (group_stats[item]["total"], rng.random()), reverse=True):
            choices = []
            for fold in range(folds):
                assignment[group_id] = fold
                choices.append((score(assignment), rng.random(), fold))
                del assignment[group_id]
            assignment[group_id] = min(choices)[2]
        current_score = score(assignment)
        if current_score < best_score:
            best, best_score = dict(assignment), current_score
    if best is None:
        raise RuntimeError("Fold assignment failed")
    fold_by_index = {index: best[str(row["group_id"])] for index, row in enumerate(rows)}
    return best, fold_by_index, best_score


@torch.inference_mode()
def predict(model, loader, device, rows):
    model.eval()
    records = []
    for images, labels, indices in loader:
        probabilities = F.softmax(model(images.to(device)), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        for position, local_index in enumerate(indices.tolist()):
            row = rows[local_index]
            records.append(
                {
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "group_id": row["group_id"],
                    "label": int(labels[position]),
                    "prediction": int(predictions[position]),
                    "predicted_class": CLASS_NAMES[int(predictions[position])],
                    "confidence": float(probabilities[position, int(predictions[position])]),
                }
            )
    return records


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])


def main() -> None:
    args = parse_args()
    clock = time.perf_counter()
    started_at = datetime.now(UTC)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "per_class_oof_metrics.json").exists():
        raise FileExistsError(f"Completed OOF output exists: {args.output_dir}")
    rows = load_split_rows(args.train_csv, args.dataset_root)
    group_assignment, fold_by_index, assignment_score = grouped_fold_assignment(rows, args.folds, args.seed)
    fold_rows = []
    for index, row in enumerate(rows):
        fold_rows.append(
            {
                "image_id": row["image_id"],
                "group_id": row["group_id"],
                "label": row["label_id"],
                "fold": fold_by_index[index] + 1,
            }
        )
    with (args.output_dir / "fold_assignments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    train_transform, evaluation_transform = build_transforms(args.image_size)
    device = torch.device(args.device)
    all_predictions = []
    fold_metrics = []
    for fold in range(args.folds):
        fold_clock = time.perf_counter()
        fold_seed = args.seed + fold + 1
        set_seed(fold_seed)
        fit = [row for index, row in enumerate(rows) if fold_by_index[index] != fold]
        holdout = [row for index, row in enumerate(rows) if fold_by_index[index] == fold]
        fit_groups = {row["group_id"] for row in fit}
        holdout_groups = {row["group_id"] for row in holdout}
        if fit_groups & holdout_groups:
            raise RuntimeError("OOF group leakage")
        fit_loader = DataLoader(
            RescueNetDataset(fit, train_transform), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.device == "cuda",
        )
        holdout_loader = DataLoader(
            RescueNetDataset(holdout, evaluation_transform), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.device == "cuda",
        )
        model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=len(CLASS_NAMES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for images, labels, _ in fit_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * labels.size(0)
                count += labels.size(0)
            scheduler.step()
            print(f"fold={fold+1}/3 epoch={epoch:03d}/{args.epochs} loss={loss_sum/count:.5f}", flush=True)
        records = predict(model, holdout_loader, device, holdout)
        for record in records:
            record["fold"] = fold + 1
        all_predictions.extend(records)
        metrics = classification_metrics(
            [int(record["label"]) for record in records],
            [int(record["prediction"]) for record in records],
        )
        fold_metrics.append(
            {
                "fold": fold + 1,
                "seed": fold_seed,
                "fit_samples": len(fit),
                "holdout_samples": len(holdout),
                "fit_groups": len(fit_groups),
                "holdout_groups": len(holdout_groups),
                "duration_seconds": time.perf_counter() - fold_clock,
                "metrics": metrics,
            }
        )
        del model

    if len(all_predictions) != len(rows) or len({record["image_id"] for record in all_predictions}) != len(rows):
        raise RuntimeError("OOF coverage is incomplete or duplicated")
    all_predictions.sort(key=lambda record: int(Path(str(record["image_id"])).stem))
    with (args.output_dir / "oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    labels = [int(record["label"]) for record in all_predictions]
    predictions = [int(record["prediction"]) for record in all_predictions]
    matrix = np.asarray(classification_metrics(labels, predictions)["confusion_matrix"], dtype=np.int64)
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64), where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    write_matrix(args.output_dir / "confusion_raw.csv", matrix)
    write_matrix(args.output_dir / "confusion_row_normalized.csv", normalized)
    pairs = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        rates = normalized[class_index].copy()
        rates[class_index] = -1
        opponent = int(np.argmax(rates))
        fallback_used = bool(np.max(rates) <= 0)
        if fallback_used:
            opponent = 1 if class_index in (0, 2) else int(0 if normalized[1, 0] >= normalized[1, 2] else 2)
        pairs.append(
            {
                "target_label": class_index,
                "target_class": class_name,
                "opponent_label": opponent,
                "opponent_class": CLASS_NAMES[opponent],
                "normalized_confusion_rate": float(normalized[class_index, opponent]),
                "raw_confusion_count": int(matrix[class_index, opponent]),
                "fallback_used": fallback_used,
                "source": "train_only_grouped_3fold_oof",
            }
        )
    write_json(args.output_dir / "confusion_pairs.json", pairs)
    metrics = classification_metrics(labels, predictions)
    audit = {
        "completion_signal": "RESCUENET_P1_OOF_COMPLETE",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - clock,
        "protocol": "train_only_grouped_3fold_oof",
        "train_split_sha256": sha256_file(args.train_csv),
        "fold_assignments_sha256": sha256_file(args.output_dir / "fold_assignments.csv"),
        "fold_count": args.folds,
        "group_count": len(group_assignment),
        "assignment_objective": assignment_score,
        "fold_metrics": fold_metrics,
        "overall_metrics": metrics,
        "validation_used": False,
        "test_used": False,
        "RemoteCLIP_used": False,
        "epochs_per_fold": args.epochs,
        "seed": args.seed,
    }
    write_json(args.output_dir / "per_class_oof_metrics.json", audit)
    (args.output_dir / "oof_audit.md").write_text(
        "# RescueNet P1 OOF Audit\n\n"
        "Completion: `RESCUENET_P1_OOF_COMPLETE`\n\n"
        f"- Samples: {len(rows)}\n- Groups: {len(group_assignment)}\n"
        f"- Accuracy: {metrics['accuracy']:.6f}\n- Macro-F1: {metrics['macro_f1']:.6f}\n"
        "- Source: grouped RescueNet train split only.\n"
        "- Validation/test/RemoteCLIP used: no.\n",
        encoding="utf-8",
    )
    print("RESCUENET_P1_OOF_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
