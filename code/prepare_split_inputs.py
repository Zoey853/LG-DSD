"""Resolve published relative split paths against the user's dataset directory."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath, PureWindowsPath


FILES = {
    "AID": ("split.csv", "oof_folds.csv"),
    "AIDER": ("train_val.csv", "test.csv", "oof_folds.csv"),
    "RescueNet": ("train.csv", "val.csv", "test.csv", "oof_folds.csv"),
}


def prepare(dataset: str, dataset_root: Path, output_dir: Path,
            manifests_root: Path | None = None) -> list[Path]:
    """Change path representation only; keep membership, labels and groups."""
    manifests_root = manifests_root or Path(__file__).resolve().parents[1] / "splits"
    dataset_root = dataset_root.expanduser().resolve()
    outputs = [output_dir / name for name in FILES[dataset]]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite existing split inputs")
    prepared = []
    for name in FILES[dataset]:
        source = manifests_root / dataset / name
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        if "relative_path" not in fields or not rows:
            raise ValueError(f"Invalid published split: {source.name}")
        if "image_path" not in fields:
            fields.append("image_path")
        for row in rows:
            raw = row["relative_path"]
            relative = PurePosixPath(raw)
            if relative.is_absolute() or PureWindowsPath(raw).drive or ".." in relative.parts:
                raise ValueError("Expected a safe dataset-relative image path")
            row["image_path"] = str(dataset_root.joinpath(*relative.parts).resolve())
        prepared.append((output_dir / name, fields, rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, fields, rows in prepared:
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FILES), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in prepare(args.dataset, args.dataset_root, args.output_dir):
        print(path.name)


if __name__ == "__main__":
    main()
