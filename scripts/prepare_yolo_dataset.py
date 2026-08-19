#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def count_files(path: Path, extensions: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.txt") if item.is_file())


def validate_split(root: Path, split: str) -> dict:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    return {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "images": count_files(image_dir, IMAGE_EXTENSIONS),
        "labels": count_labels(label_dir),
        "ok": image_dir.exists() and label_dir.exists() and count_files(image_dir, IMAGE_EXTENSIONS) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO-format blister dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing train/valid/test folders.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    splits = [validate_split(root, split) for split in ("train", "valid", "val", "test")]
    summary = {
        "root": str(root),
        "splits": splits,
        "ready_for_training": any(item["split"] == "train" and item["ok"] for item in splits)
        and any(item["split"] in {"valid", "val"} and item["ok"] for item in splits),
    }
    output = args.out or root / "visionops_yolo_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

