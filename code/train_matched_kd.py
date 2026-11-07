"""Paper KD for AIDER/RescueNet: original training-function extraction.

The training body is unchanged. Only imports and runtime path defaults are
adapted for this partial release. No scheduler or automatic Test evaluation.
Call train_matched_b0(dataset, seed, method) after supplying external inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import train_unified_oof_semantic as u

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "outputs" / "paper_kd"

def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def train_matched_b0(dataset, seed, method):
    """Use the unified B1/B2 data and optimizer protocol with semantic loss zero."""
    folder = HERE / "runs" / f"{dataset}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    detail = method["dataset_details"][dataset]
    inputs = detail["inputs"]
    spec = argparse.Namespace(
        dataset=dataset,
        split_file=Path(inputs["split_file"]["path"]) if dataset == "AIDER" else None,
        train_csv=Path(inputs["train_csv"]["path"]) if dataset == "RescueNet" else None,
        val_csv=Path(inputs["val_csv"]["path"]) if dataset == "RescueNet" else None,
        dataset_root=ROOT / "RescueNet" if dataset == "RescueNet" else None,
    )
    u.set_seed(seed)
    rows, names, split_hashes = u.load_rows(spec)
    teacher, teacher_meta = u.load_teacher(Path(inputs["class_teacher_cache"]["path"]), rows, dataset)
    train_transform, val_transform = u.build_transforms()
    train_loader = DataLoader(u.UnifiedDataset(rows["train"], train_transform, teacher, dataset),
                              batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(u.UnifiedDataset(rows["val"], val_transform),
                            batch_size=32, shuffle=False, num_workers=0)
    model = u.Student(len(names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    best_pair, best_epoch = (-1.0, -1.0), 0
    best_path = folder / "best.pt"
    history = []
    started = time.perf_counter()
    for epoch in range(1, 51):
        epoch_start = time.perf_counter()
        model.train()
        sums = defaultdict(float)
        count = 0
        for images, labels, _, teacher_logits in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            ce = F.cross_entropy(logits, labels)
            kd = F.kl_div(F.log_softmax(logits / 8.0, dim=1),
                          F.softmax(teacher_logits / 8.0, dim=1), reduction="batchmean") * 64.0
            loss = 0.9 * ce + 0.1 * kd
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite B0 loss")
            loss.backward()
            optimizer.step()
            size = len(labels)
            count += size
            for key, value in (("loss", loss), ("ce", ce), ("kd", kd)):
                sums[key] += float(value.detach()) * size
        scheduler.step()
        validation, _ = u.evaluate(model, val_loader, torch.device("cpu"), rows["val"], names)
        candidate = (validation["macro_f1"], validation["accuracy"])
        if candidate > best_pair:
            best_pair, best_epoch = candidate, epoch
            torch.save({"epoch": epoch, "class_names": names,
                        "model_state_dict": model.state_dict(), "student_only": model.student_state(),
                        "validation": validation,
                        "excluded_training_only_components": ["projector", "RemoteCLIP", "teacher_cache"]}, best_path)
        history.append({"epoch": epoch, "duration_seconds": time.perf_counter() - epoch_start,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train": {key: value / count for key, value in sums.items()},
                        "validation": {key: validation[key] for key in ("accuracy", "macro_f1")}})
        print(f"B0_ALIGNED {dataset} seed={seed} epoch={epoch}/50 "
              f"val_acc={validation['accuracy']:.6f} val_f1={validation['macro_f1']:.6f} "
              f"best_epoch={best_epoch}", flush=True)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    measured, predictions = u.evaluate(model, val_loader, torch.device("cpu"), rows["val"], names)
    if abs(measured["macro_f1"] - checkpoint["validation"]["macro_f1"]) > 1e-12:
        raise ValueError("Best-checkpoint validation replay changed")
    prediction_path = folder / "validation_predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    result = {
        "status": "COMPLETE", "dataset": dataset, "variant": "B0_ALIGNED", "seed": seed,
        "best_epoch": best_epoch, "best_validation": measured, "history": history,
        "configuration": {"student": "MobileNetV3-Small", "epochs": 50, "batch_size": 32,
                          "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
                          "kd_alpha": 0.1, "kd_temperature": 8.0, "semantic_weight": 0,
                          "augmentation": "RandomResizedCrop(224, scale=(0.7,1.0)); RandomHorizontalFlip; no RandomVerticalFlip",
                          "checkpoint_selection": "validation Macro-F1, Accuracy tie-break, earliest exact tie"},
        "input_hashes": {**split_hashes, "class_teacher_cache_sha256": sha(inputs["class_teacher_cache"]["path"])},
        "teacher_metadata_test_used": teacher_meta.get("test_used", teacher_meta.get("test_data_used", False)),
        "checkpoint": record_ref(best_path), "validation_predictions": record_ref(prediction_path),
        "runtime_seconds": time.perf_counter() - started, "test_used": False, "test_images_read": 0,
    }
    write(folder / "run_summary.json", result)
    return result
