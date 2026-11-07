"""Train a MobileNetV3 student with CE or CLIP semantic distillation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_FILE = CODE_DIR / "splits" / "aid_seed42_trainpool50.csv"
DEFAULT_TEACHER_CACHE = (
    CODE_DIR
    / "outputs"
    / "teacher_soft_labels_aid_full"
    / "clip_ViT-B-32_class_name_prompt.pt"
)
DEFAULT_OUTPUT_DIR = CODE_DIR / "outputs" / "student_aid"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


class AIDStudentDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        transform,
        teacher_by_path: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.teacher_by_path = teacher_by_path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = str(row["image_path"])
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = int(row["label"])
        if self.teacher_by_path is None:
            return image_tensor, label
        teacher_logits = self.teacher_by_path[normalize_path(image_path)]
        return image_tensor, label, teacher_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on AID.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["ce", "kd"], required=True)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument(
        "--kd-warmup-epochs",
        type=int,
        default=0,
        help="Train with CE only for this many epochs before enabling KD.",
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--teacher-logit-scale", type=float, default=1.0)
    parser.add_argument(
        "--kd-gate",
        choices=["none", "teacher_correct"],
        default="none",
        help="Optionally apply KD only when the teacher predicts the ground-truth class.",
    )
    parser.add_argument(
        "--kd-weighting",
        choices=["none", "entropy"],
        default="none",
        help="Optionally weight each sample's KD loss by teacher reliability.",
    )
    parser.add_argument(
        "--kd-weight-floor",
        type=float,
        default=0.05,
        help="Minimum raw reliability before mean-one normalization.",
    )
    parser.add_argument(
        "--kd-incorrect-weight",
        type=float,
        default=0.0,
        help="KD multiplier for teacher-incorrect samples when correctness gating is active.",
    )
    parser.add_argument(
        "--kd-agreement-weighting",
        choices=["none", "cosine"],
        default="none",
        help="Optionally weight KD by detached teacher-student probability agreement.",
    )
    parser.add_argument(
        "--kd-agreement-floor",
        type=float,
        default=0.1,
        help="Minimum dynamic agreement multiplier.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Select hyperparameters using validation only; do not evaluate the test split.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "macro_f1"],
        default="accuracy",
        help="Validation metric used for checkpoint selection; the other metric breaks ties.",
    )
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    if not 0 <= args.kd_alpha <= 1:
        parser.error("--kd-alpha must be between 0 and 1.")
    if args.kd_warmup_epochs < 0:
        parser.error("--kd-warmup-epochs must be non-negative.")
    if args.kd_warmup_epochs >= args.epochs:
        parser.error("--kd-warmup-epochs must be smaller than --epochs.")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")
    if args.teacher_logit_scale <= 0:
        parser.error("--teacher-logit-scale must be greater than zero.")
    if not 0 <= args.kd_weight_floor <= 1:
        parser.error("--kd-weight-floor must be between zero and one.")
    if not 0 <= args.kd_incorrect_weight <= 1:
        parser.error("--kd-incorrect-weight must be between zero and one.")
    if not 0 <= args.kd_agreement_floor <= 1:
        parser.error("--kd-agreement-floor must be between zero and one.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_rows(
    split_file: Path, max_samples_per_class: int | None
) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
    with split_file.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Split file is empty: {split_file}")

    class_names_by_label = {
        int(row["label"]): str(row["class_name"]) for row in rows
    }
    class_names = [class_names_by_label[label] for label in sorted(class_names_by_label)]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    per_split_class_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        split = str(row["split"])
        label = int(row["label"])
        if (
            max_samples_per_class is not None
            and per_split_class_counts[(split, label)] >= max_samples_per_class
        ):
            continue
        grouped[split].append(row)
        per_split_class_counts[(split, label)] += 1
    return grouped, class_names


def load_teacher_logits(cache_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = payload["image_paths"]
    teacher_logits = payload["teacher_logits"].float()
    if len(image_paths) != len(teacher_logits):
        raise ValueError("Teacher cache paths and logits have different lengths.")
    teacher_by_path = {
        normalize_path(image_path): teacher_logits[index]
        for index, image_path in enumerate(image_paths)
    }
    return teacher_by_path, payload["metadata"]


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
    kd_alpha: float,
    temperature: float,
    teacher_logit_scale: float,
    kd_gate: str,
    kd_weighting: str,
    kd_weight_floor: float,
    kd_incorrect_weight: float,
    kd_agreement_weighting: str,
    kd_agreement_floor: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_reliability = 0.0
    total_agreement = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(images)
        ce_loss = F.cross_entropy(student_logits, labels)
        reliability = torch.ones(labels.shape[0], device=device)
        agreement = torch.ones_like(reliability)

        if mode == "kd":
            teacher_logits = batch[2].to(device) * teacher_logit_scale
            kd_loss_per_class = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="none",
            )
            kd_loss_per_sample = kd_loss_per_class.sum(dim=1) * (temperature**2)
            if kd_weighting == "entropy":
                teacher_probabilities = F.softmax(teacher_logits, dim=1)
                normalized_entropy = -(
                    teacher_probabilities
                    * teacher_probabilities.clamp_min(1e-12).log()
                ).sum(dim=1) / torch.log(
                    torch.tensor(
                        teacher_logits.shape[1],
                        dtype=teacher_logits.dtype,
                        device=teacher_logits.device,
                    )
                )
                reliability = reliability * (1 - normalized_entropy).clamp_min(
                    kd_weight_floor
                )
            if kd_gate == "teacher_correct":
                teacher_correct = teacher_logits.argmax(dim=1).eq(labels)
                correctness_weight = torch.where(
                    teacher_correct,
                    torch.ones_like(reliability),
                    torch.full_like(reliability, kd_incorrect_weight),
                )
                reliability = reliability * correctness_weight
            if kd_agreement_weighting == "cosine":
                teacher_probabilities = F.softmax(teacher_logits, dim=1)
                student_probabilities = F.softmax(student_logits.detach(), dim=1)
                agreement = F.cosine_similarity(
                    teacher_probabilities, student_probabilities, dim=1
                ).clamp_min(kd_agreement_floor)
                reliability = reliability * agreement
            else:
                agreement = torch.ones_like(reliability)
            if (
                kd_gate != "none"
                or kd_weighting != "none"
                or kd_agreement_weighting != "none"
            ):
                kd_loss = (kd_loss_per_sample * reliability).sum() / reliability.sum().clamp_min(
                    1e-12
                )
            else:
                kd_loss = kd_loss_per_sample.mean()
            loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
        else:
            kd_loss = torch.zeros((), device=device)
            loss = ce_loss

        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_kd += kd_loss.item() * batch_size
        total_reliability += reliability.detach().mean().item() * batch_size
        total_agreement += agreement.detach().mean().item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "kd_loss": total_kd / total_samples,
        "teacher_reliability_raw_mean": total_reliability / total_samples,
        "teacher_student_agreement_mean": total_agreement / total_samples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    rows_by_split, class_names = load_split_rows(
        args.split_file, args.max_samples_per_class
    )
    teacher_by_path = None
    teacher_metadata = None
    if args.mode == "kd":
        teacher_by_path, teacher_metadata = load_teacher_logits(args.teacher_cache)
        missing_paths = [
            str(row["image_path"])
            for row in rows_by_split["train"]
            if normalize_path(str(row["image_path"])) not in teacher_by_path
        ]
        if missing_paths:
            raise ValueError(f"Teacher cache is missing {len(missing_paths)} training images.")

    train_transform, evaluation_transform = build_transforms(args.image_size)
    datasets = {
        "train": AIDStudentDataset(rows_by_split["train"], train_transform, teacher_by_path),
        "val": AIDStudentDataset(rows_by_split["val"], evaluation_transform),
        "test": AIDStudentDataset(rows_by_split["test"], evaluation_transform),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    model = timm.create_model(
        args.model_name, pretrained=args.pretrained, num_classes=len(class_names)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    initialization = "pretrained" if args.pretrained else "scratch"
    if args.mode == "kd":
        alpha_tag = str(args.kd_alpha).replace(".", "p")
        temperature_tag = str(args.temperature).replace(".", "p")
        scale_tag = str(args.teacher_logit_scale).replace(".", "p")
        gate_tag = "" if args.kd_gate == "none" else f"_g{args.kd_gate}"
        incorrect_tag = (
            ""
            if args.kd_gate == "none"
            else f"_beta{str(args.kd_incorrect_weight).replace('.', 'p')}"
        )
        weighting_tag = (
            "" if args.kd_weighting == "none" else f"_weight-{args.kd_weighting}"
        )
        agreement_tag = (
            ""
            if args.kd_agreement_weighting == "none"
            else f"_agree-{args.kd_agreement_weighting}"
        )
        warmup_tag = "" if args.kd_warmup_epochs == 0 else f"_w{args.kd_warmup_epochs}"
        run_name = (
            f"{args.model_name}_kd_{initialization}_a{alpha_tag}_t{temperature_tag}"
            f"_s{scale_tag}{gate_tag}{incorrect_tag}{weighting_tag}{agreement_tag}"
            f"{warmup_tag}_seed{args.seed}"
        )
    else:
        run_name = f"{args.model_name}_ce_{initialization}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode.upper()} | Model: {args.model_name} | Device: {device}")
    print(
        "Samples: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )
    history: list[dict[str, object]] = []
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    checkpoint_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_mode = (
            "ce"
            if args.mode == "kd" and epoch <= args.kd_warmup_epochs
            else args.mode
        )
        effective_kd_alpha = 0.0 if epoch_mode == "ce" else args.kd_alpha
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            epoch_mode,
            effective_kd_alpha,
            args.temperature,
            args.teacher_logit_scale,
            args.kd_gate,
            args.kd_weighting,
            args.kd_weight_floor,
            args.kd_incorrect_weight,
            args.kd_agreement_weighting,
            args.kd_agreement_floor,
        )
        validation_metrics = evaluate(model, loaders["val"], device)
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "effective_kd_alpha": effective_kd_alpha,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d}: train_loss={train_metrics['loss']:.4f}, "
            f"val_acc={validation_metrics['accuracy']:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        candidate = (
            validation_metrics[args.selection_metric],
            validation_metrics["accuracy" if args.selection_metric == "macro_f1" else "macro_f1"],
        )
        incumbent = (
            best_val_macro_f1 if args.selection_metric == "macro_f1" else best_val_accuracy,
            best_val_accuracy if args.selection_metric == "macro_f1" else best_val_macro_f1,
        )
        if candidate > incumbent:
            best_val_accuracy = validation_metrics["accuracy"]
            best_val_macro_f1 = validation_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "class_names": class_names,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = None if args.skip_test else evaluate(model, loaders["test"], device)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "split_file": str(args.split_file.resolve()),
        "sample_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "teacher_cache": str(args.teacher_cache.resolve()) if args.mode == "kd" else None,
        "teacher_metadata": teacher_metadata,
        "kd_alpha": args.kd_alpha if args.mode == "kd" else None,
        "kd_warmup_epochs": args.kd_warmup_epochs if args.mode == "kd" else None,
        "temperature": args.temperature if args.mode == "kd" else None,
        "teacher_logit_scale": (
            args.teacher_logit_scale if args.mode == "kd" else None
        ),
        "kd_gate": args.kd_gate if args.mode == "kd" else None,
        "kd_weighting": args.kd_weighting if args.mode == "kd" else None,
        "kd_weight_floor": args.kd_weight_floor if args.mode == "kd" else None,
        "kd_incorrect_weight": (
            args.kd_incorrect_weight if args.mode == "kd" else None
        ),
        "kd_agreement_weighting": (
            args.kd_agreement_weighting if args.mode == "kd" else None
        ),
        "kd_agreement_floor": (
            args.kd_agreement_floor if args.mode == "kd" else None
        ),
        "test_evaluated": not args.skip_test,
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_macro_f1": best_val_macro_f1,
        "selection_metric": args.selection_metric,
        "test": test_metrics,
        "history": history,
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, default=str)
    if test_metrics is None:
        print("Test evaluation skipped for validation-only model selection.")
    else:
        print(
            f"Test: Acc={test_metrics['accuracy']:.4f}, "
            f"Macro-F1={test_metrics['macro_f1']:.4f}"
        )
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
