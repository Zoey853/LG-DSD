"""Prepare the frozen AID Top-1 transfer inputs from existing train-only OOF outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OOF_DIR = HERE / "outputs" / "aid_train_oof_confusions_v1"
OUTPUT = HERE / "outputs" / "aid_top1_fixed_residual_transfer"
SPLIT = HERE / "splits" / "aid_seed42_trainpool50.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summary_path = OOF_DIR / "summary.json"
    counts_path = OOF_DIR / "confusion_counts.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_split") != "train_only":
        raise ValueError("OOF source is not train-only")
    if summary.get("validation_split_used") or summary.get("test_split_used"):
        raise ValueError("OOF used validation or test data")
    if int(summary.get("num_train_samples", -1)) != int(summary.get("num_oof_predictions", -2)):
        raise ValueError("OOF coverage is incomplete")
    class_names = list(summary["class_names"])
    with counts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0][1:] != class_names or [row[0] for row in rows[1:]] != class_names:
        raise ValueError("Confusion matrix class order differs from OOF summary")
    matrix = [[int(value) for value in row[1:]] for row in rows[1:]]
    pairs, prompt_classes = [], []
    for label, class_name in enumerate(class_names):
        candidates = [(count, opponent) for opponent, count in enumerate(matrix[label]) if opponent != label]
        raw_count, opponent = max(candidates, key=lambda item: (item[0], -item[1]))
        fallback_used = raw_count == 0
        if fallback_used:
            # Preserve an OOF-derived opponent when a class has no directed OOF error:
            # use the strongest bidirectional pair already recorded by the same OOF audit.
            recorded = summary["confusable_classes"][label]["top_confusable"][0]
            opponent = int(recorded["label"])
        total = sum(matrix[label])
        readable = class_name.replace("Residential", " residential").replace("Land", " land")
        readable = " ".join(readable.replace("_", " ").split()).lower()
        opponent_readable = class_names[opponent].replace("Residential", " residential").replace("Land", " land")
        opponent_readable = " ".join(opponent_readable.replace("_", " ").split()).lower()
        pair = {
            "target_label": label,
            "target_class": class_name,
            "opponent_label": opponent,
            "opponent_class": class_names[opponent],
            "normalized_confusion_rate": raw_count / total if total else 0.0,
            "raw_confusion_count": raw_count,
            "fallback_used": fallback_used,
            "source": "train_only_stratified_5fold_oof",
        }
        pairs.append(pair)
        prompt_classes.append({
            "class_name": class_name,
            "readable_name": readable,
            "label": label,
            "class_name_prompt": f"A remote sensing image of {readable}.",
            "confusable_classes": [opponent_readable],
            "top1_confusing_class": class_names[opponent],
            "oof_confusion_rate": pair["normalized_confusion_rate"],
            "oof_confusion_count": raw_count,
            "fallback_used": fallback_used,
            "source": pair["source"],
            "task_context": "AID aerial remote-sensing scene classification",
        })
    write_json(OUTPUT / "confusion_pairs.json", pairs)
    write_json(OUTPUT / "prompt_input.json", {
        "metadata": {
            "dataset_name": "AID",
            "task": "AID aerial remote-sensing scene classification",
            "source": "existing train-only stratified 5-fold OOF",
            "oof_summary": str(summary_path.resolve()),
            "oof_summary_sha256": sha256(summary_path),
            "confusion_counts_sha256": sha256(counts_path),
            "split_sha256": sha256(SPLIT),
            "validation_used": False,
            "test_used": False,
        },
        "classes": prompt_classes,
    })
    write_json(OUTPUT / "preflight_audit.json", {
        "completion_signal": "AID_TOP1_TRANSFER_PREFLIGHT_COMPLETE",
        "decision": "PASS",
        "new_oof_training": False,
        "oof_reused": str(summary_path.resolve()),
        "top1_definition": "highest directed row-normalized train-only OOF confusion; class-index tie break",
        "class_count": len(class_names),
        "one_opponent_per_class": len(pairs) == len(class_names),
        "validation_used": False,
        "test_used": False,
    })
    print("AID_TOP1_TRANSFER_PREFLIGHT_COMPLETE")


if __name__ == "__main__":
    main()
