"""Train U1/U2/U3 under one frozen, validation-only protocol.

U0 is deliberately out of scope: this program never creates, rewrites, or
otherwise modifies a U0 result. RemoteCLIP prototypes and OOF pairs are used
only while training; the saved checkpoint contains the student alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("AID", "AIDER", "RescueNet"), required=True)
    parser.add_argument("--line", choices=("U1", "U2", "U3"), required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--class-teacher-cache", type=Path, required=True)
    parser.add_argument("--semantic-prototype-cache", type=Path, required=True)
    parser.add_argument("--confusion-pairs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--semantic-weight", type=float, default=0.1)
    parser.add_argument("--semantic-temperature", type=float, default=0.07)
    parser.add_argument("--hard-margin-tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.dataset in ("AID", "AIDER") and not args.split_file:
        parser.error("AID/AIDER require --split-file")
    if args.dataset == "RescueNet" and not (args.train_csv and args.val_csv and args.dataset_root):
        parser.error("RescueNet requires --train-csv, --val-csv, and --dataset-root")
    for path in (args.class_teacher_cache, args.semantic_prototype_cache, args.confusion_pairs):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.epochs != 50 or args.batch_size != 32:
        parser.error("Frozen protocol requires 50 epochs and batch size 32")
    if not np.isclose(args.learning_rate, 3e-4) or not np.isclose(args.weight_decay, 1e-4):
        parser.error("Frozen optimizer is AdamW lr=3e-4, weight_decay=1e-4")
    if not np.isclose(args.temperature, 8.0):
        parser.error("Frozen KD temperature is 8.0")
    if not any(np.isclose(args.semantic_weight, value) for value in (0.03, 0.05, 0.1)):
        parser.error("semantic_weight must be one of 0.03, 0.05, or 0.10")
    expected_alpha = 0.25 if args.dataset == "AID" else 0.1
    if not np.isclose(args.kd_alpha, expected_alpha):
        parser.error(f"{args.dataset} frozen class-name KD alpha is {expected_alpha}")
    if not np.isclose(args.semantic_temperature, 0.07) or not np.isclose(args.hard_margin_tau, 0.1):
        parser.error("Unified semantic temperature/tau are 0.07/0.1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.output_json.exists() or args.output_predictions.exists():
        parser.error("Refusing to overwrite a completed/partial formal run")
    return args


def normalize_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rows(args: argparse.Namespace):
    if args.dataset in ("AID", "AIDER"):
        with args.split_file.open("r", encoding="utf-8-sig", newline="") as handle:
            source = list(csv.DictReader(handle))
        rows = defaultdict(list)
        class_by_label = {}
        for item in source:
            split = str(item["split"])
            if split not in ("train", "val"):
                continue
            label = int(item["label"])
            class_by_label[label] = str(item["class_name"])
            rows[split].append({
                "image_path": str(Path(item["image_path"]).resolve()),
                "sample_id": item.get("relative_path") or Path(item["image_path"]).name,
                "group_id": item.get("group_id", ""),
                "label": label,
            })
        class_names = [class_by_label[index] for index in range(len(class_by_label))]
        split_hashes = {"split_file_sha256": sha256(args.split_file)}
    else:
        rows = defaultdict(list)
        class_by_label = {}
        for split, csv_path in (("train", args.train_csv), ("val", args.val_csv)):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for item in csv.DictReader(handle):
                    label = int(item["label_id"])
                    class_by_label[label] = str(item["label"])
                    rows[split].append({
                        "image_path": str((args.dataset_root / item["relative_path"]).resolve()),
                        "sample_id": str(item["image_id"]),
                        "group_id": str(item["group_id"]),
                        "label": label,
                    })
        class_names = [class_by_label[index] for index in range(len(class_by_label))]
        split_hashes = {"train_csv_sha256": sha256(args.train_csv), "val_csv_sha256": sha256(args.val_csv)}
    if not rows["train"] or not rows["val"]:
        raise ValueError("Train/validation rows are incomplete")
    train_groups = {row["group_id"] for row in rows["train"] if row["group_id"]}
    val_groups = {row["group_id"] for row in rows["val"] if row["group_id"]}
    if train_groups & val_groups:
        raise ValueError("Train/validation group overlap")
    return rows, class_names, split_hashes


def certify_cache(metadata: dict) -> None:
    if metadata.get("test_data_used") is True or metadata.get("test_used") is True:
        raise ValueError("Cache reports test use")
    if int(metadata.get("test_images_read", 0)) != 0:
        raise ValueError("Cache reports test image reads")
    included = set(metadata.get("included_splits", []))
    if included and included - {"train", "val"}:
        raise ValueError("Cache includes a forbidden split")


def load_teacher(path: Path, rows, dataset: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    certify_cache(metadata)
    logits = payload["teacher_logits"].float()
    if dataset == "RescueNet":
        keys = [str(value) for value in payload["image_ids"]]
        requested = {row["sample_id"] for split in ("train", "val") for row in rows[split]}
        teacher = {key: logits[index] for index, key in enumerate(keys) if key in requested}
    else:
        keys = [normalize_path(value) for value in payload["image_paths"]]
        requested = {normalize_path(row["image_path"]) for split in ("train", "val") for row in rows[split]}
        teacher = {key: logits[index] for index, key in enumerate(keys) if key in requested}
    missing = []
    for split in ("train", "val"):
        for row in rows[split]:
            key = row["sample_id"] if dataset == "RescueNet" else normalize_path(row["image_path"])
            if key not in teacher:
                missing.append(key)
    if missing:
        raise ValueError(f"Teacher cache missing {len(missing)} train/validation samples")
    return teacher, metadata


def load_prototypes(path: Path, class_names: list[str]) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    certify_cache(metadata)
    prototypes = payload["text_prototypes"]
    if isinstance(prototypes, dict):
        prototypes = prototypes["discriminative_attribute_ensemble"]
    prototypes = F.normalize(prototypes.float(), dim=1)
    if prototypes.shape != (len(class_names), 512):
        raise ValueError(f"Expected {len(class_names)}x512 semantic prototypes, got {tuple(prototypes.shape)}")
    return prototypes, metadata


def load_pairs(path: Path, class_names: list[str]) -> torch.Tensor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("pairs", payload.get("confusion_pairs", payload))
    if not isinstance(payload, list) or len(payload) != len(class_names):
        raise ValueError("Expected exactly one OOF opponent per class")
    pairs = torch.empty((len(class_names), 2), dtype=torch.long)
    seen = set()
    for item in payload:
        target = int(item["target_label"])
        opponent = int(item["opponent_label"])
        source = str(item.get("source", "")).lower()
        if "train_only" not in source or "oof" not in source:
            raise ValueError("Pair is not certified train-only OOF")
        if target == opponent or target in seen or not 0 <= opponent < len(class_names):
            raise ValueError("Invalid OOF pair")
        pairs[target] = torch.tensor([target, opponent])
        seen.add(target)
    if seen != set(range(len(class_names))):
        raise ValueError("OOF pairs do not cover every class")
    return pairs


def build_transforms():
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([
        transforms.Resize(round(224 / 0.875)), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


class UnifiedDataset(Dataset):
    def __init__(self, rows, transform, teacher=None, dataset="AID"):
        self.rows = rows
        self.transform = transform
        self.teacher = teacher
        self.dataset = dataset

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row["label"])
        if self.teacher is None:
            return tensor, label, index
        key = row["sample_id"] if self.dataset == "RescueNet" else normalize_path(row["image_path"])
        return tensor, label, index, self.teacher[key]


class Student(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0)
        # timm's MobileNetV3 ``num_features`` describes the pre-conv-head
        # width (576), while ``forward_head(..., pre_logits=True)`` returns
        # the pooled conv-head representation (1024).
        feature_dim = int(self.backbone.conv_head.out_channels)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 512))

    def forward(self, images):
        maps = self.backbone.forward_features(images)
        features = self.backbone.forward_head(maps, pre_logits=True)
        return self.classifier(features), F.normalize(self.projector(features), dim=1)

    def student_state(self):
        return {"backbone_state_dict": self.backbone.state_dict(), "classifier_state_dict": self.classifier.state_dict()}


def metrics(labels, predictions, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device, rows, class_names):
    model.eval()
    labels, predictions, probabilities, sample_ids, group_ids = [], [], [], [], []
    for images, batch_labels, indices in loader:
        logits, _ = model(images.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        preds = probs.argmax(1)
        for position, index in enumerate(indices.tolist()):
            labels.append(int(batch_labels[position]))
            predictions.append(int(preds[position]))
            probabilities.append(probs[position].numpy())
            sample_ids.append(rows[index]["sample_id"])
            group_ids.append(rows[index]["group_id"])
    return metrics(labels, predictions, class_names), {
        "sample_id": np.asarray(sample_ids), "group_id": np.asarray(group_ids),
        "label": np.asarray(labels, dtype=np.int64), "prediction": np.asarray(predictions, dtype=np.int64),
        "probabilities": np.stack(probabilities).astype(np.float32),
    }


def semantic_loss(line, projected, logits, labels, pairs, prototypes, temperature, tau):
    pair_indices = pairs[labels]
    target_prototypes = prototypes[pair_indices]
    if line == "U1":
        pair_logits = torch.einsum("bd,bkd->bk", projected, target_prototypes) / temperature
        per_sample = F.cross_entropy(pair_logits, torch.zeros_like(labels), reduction="none")
        mask = torch.ones_like(per_sample, dtype=torch.bool)
    else:
        directions = F.normalize(target_prototypes[:, 0] - target_prototypes[:, 1], dim=1)
        per_sample = 1.0 - F.cosine_similarity(projected, directions, dim=1)
        if line == "U3":
            probs = F.softmax(logits, dim=1)
            true_p = probs.gather(1, labels[:, None]).squeeze(1)
            opponent_p = probs.gather(1, pair_indices[:, 1, None]).squeeze(1)
            mask = (true_p - opponent_p) < tau
        else:
            mask = torch.ones_like(per_sample, dtype=torch.bool)
    loss = per_sample[mask].mean() if bool(mask.any()) else projected.sum() * 0.0
    return loss, int(mask.sum().detach()), int(mask.numel())


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    clock = time.perf_counter()
    set_seed(args.seed)
    rows, class_names, split_hashes = load_rows(args)
    teacher, teacher_meta = load_teacher(args.class_teacher_cache, rows, args.dataset)
    prototypes_cpu, prototype_meta = load_prototypes(args.semantic_prototype_cache, class_names)
    pairs_cpu = load_pairs(args.confusion_pairs, class_names)
    train_transform, val_transform = build_transforms()
    train_loader = DataLoader(
        UnifiedDataset(rows["train"], train_transform, teacher, args.dataset), batch_size=32,
        shuffle=True, num_workers=args.num_workers, pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        UnifiedDataset(rows["val"], val_transform), batch_size=32,
        shuffle=False, num_workers=args.num_workers, pin_memory=args.device == "cuda",
    )
    device = torch.device(args.device)
    model = Student(len(class_names)).to(device)
    prototypes, pairs = prototypes_cpu.to(device), pairs_cpu.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_epoch, best_pair, best_metrics = 0, (-1.0, -1.0), None
    history = []
    for epoch in range(1, 51):
        epoch_clock = time.perf_counter()
        model.train()
        sums = defaultdict(float)
        sample_count = active_count = 0
        for images, labels, _, teacher_logits in train_loader:
            images, labels, teacher_logits = images.to(device), labels.to(device), teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits, projected = model(images)
            ce = F.cross_entropy(student_logits, labels)
            kd = F.kl_div(
                F.log_softmax(student_logits / 8.0, dim=1), F.softmax(teacher_logits / 8.0, dim=1),
                reduction="batchmean",
            ) * 64.0
            semantic, active, total = semantic_loss(
                args.line, projected, student_logits, labels, pairs, prototypes, 0.07, 0.1
            )
            loss = (1.0 - args.kd_alpha) * ce + args.kd_alpha * kd + args.semantic_weight * semantic
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            optimizer.step()
            count = len(labels)
            sample_count += count
            active_count += active
            for key, value in (("loss", loss), ("ce", ce), ("kd", kd), ("semantic", semantic)):
                sums[key] += float(value.detach()) * count
        scheduler.step()
        validation, _ = evaluate(model, val_loader, device, rows["val"], class_names)
        candidate = (validation["macro_f1"], validation["accuracy"])
        if candidate > best_pair:
            best_pair, best_epoch, best_metrics = candidate, epoch, validation
            torch.save({
                "epoch": epoch, "class_names": class_names, "model_state_dict": model.state_dict(),
                "student_only": model.student_state(), "validation": validation,
                "excluded_training_only_components": ["projector", "RemoteCLIP", "text_prototypes", "OOF_pairs", "teacher_cache"],
            }, args.checkpoint)
        history.append({
            "epoch": epoch, "duration_seconds": time.perf_counter() - epoch_clock,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / sample_count for key, value in sums.items()},
            "semantic_active_fraction": active_count / sample_count,
            "validation": validation,
        })
        print(
            f"{args.line} {args.dataset} seed={args.seed} epoch={epoch:02d}/50 "
            f"acc={validation['accuracy']:.6f} macro_f1={validation['macro_f1']:.6f} best={best_epoch}",
            flush=True,
        )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_metrics, predictions = evaluate(model, val_loader, device, rows["val"], class_names)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_predictions, **predictions)
    result = {
        "schema_version": "1.0.0", "completion_signal": "UNIFIED_OOF_SEMANTIC_RUN_COMPLETE",
        "dataset": args.dataset, "line": args.line, "seed": args.seed,
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - clock,
        "configuration": {
            "student": "ImageNet-pretrained MobileNetV3-Small", "epochs": 50, "batch_size": 32,
            "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
            "kd_alpha": args.kd_alpha, "kd_temperature": 8.0, "semantic_weight": args.semantic_weight,
            "semantic_temperature": 0.07, "hard_margin_tau": 0.1 if args.line == "U3" else None,
            "checkpoint_selection": "validation Macro-F1, Accuracy tie-break",
        },
        "loss_definition": {
            "U1": "pair-only CE over cosine(projected_student, [t_y_diff,t_j_diff])",
            "U2": "1-cosine(projected_student, normalize(t_y_diff-t_j_diff))",
            "U3": "U2 activated only when p(y)-p(j)<0.1",
        }[args.line],
        "sample_counts": {key: len(rows[key]) for key in ("train", "val")}, "class_names": class_names,
        "best_epoch": best_epoch, "best_validation": best_metrics, "history": history,
        "oof_pairs": pairs_cpu.tolist(), "prediction_file": str(args.output_predictions.resolve()),
        "input_hashes": {
            **split_hashes, "class_teacher_cache_sha256": sha256(args.class_teacher_cache),
            "semantic_prototype_cache_sha256": sha256(args.semantic_prototype_cache),
            "confusion_pairs_sha256": sha256(args.confusion_pairs),
        },
        "cache_metadata": {"class_teacher": teacher_meta, "semantic_prototypes": prototype_meta},
        "inference": {"student_only": True, "added_inference_cost": False},
        "test_used": False, "test_images_read": 0,
    }
    write_json(args.output_json, result)
    print("UNIFIED_OOF_SEMANTIC_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
