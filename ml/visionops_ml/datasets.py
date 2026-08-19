from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    split: str
    category: str
    defect_type: str


def list_images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_simple_anomaly_dataset(root: Path) -> list[ImageRecord]:
    """Load a generic good/defect folder.

    Expected layout:

    root/
      good/*.png
      defect/*.png
    """

    records: list[ImageRecord] = []
    good_images = list_images(root / "good")
    train_count = len(good_images)
    if len(good_images) >= 8:
        train_count = max(3, int(len(good_images) * 0.75))
    for index, path in enumerate(good_images):
        split = "train" if index < train_count else "test"
        records.append(ImageRecord(path=path, label=0, split=split, category=root.name, defect_type="good"))
    for path in list_images(root / "defect"):
        records.append(ImageRecord(path=path, label=1, split="test", category=root.name, defect_type=path.stem.split("_")[0]))
    return records


def load_mvtec_category(root: Path, category: str) -> list[ImageRecord]:
    category_root = root / category
    records: list[ImageRecord] = []
    for split in ("train", "test"):
        split_root = category_root / split
        if not split_root.exists():
            continue
        for defect_dir in sorted(item for item in split_root.iterdir() if item.is_dir()):
            label = 0 if defect_dir.name == "good" else 1
            for path in list_images(defect_dir):
                records.append(
                    ImageRecord(
                        path=path,
                        label=label,
                        split=split,
                        category=category,
                        defect_type=defect_dir.name,
                    )
                )
    return records


def load_visa_category(root: Path, category: str) -> list[ImageRecord]:
    """Load VisA after extraction.

    The official tar includes `split_csv/1cls.csv`; use that split when
    present. A fallback supports common train/test good/bad layouts.
    """

    split_csv = root / "split_csv" / "1cls.csv"
    if split_csv.exists():
        records: list[ImageRecord] = []
        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("object") != category:
                    continue
                label = row.get("label", "").lower()
                image = row.get("image", "")
                if not image:
                    continue
                records.append(
                    ImageRecord(
                        path=root / image,
                        label=0 if label == "normal" else 1,
                        split=row.get("split", "test"),
                        category=category,
                        defect_type="good" if label == "normal" else "anomaly",
                    )
                )
        if records:
            return records

    candidates = [path for path in root.rglob(category) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find VisA category {category!r} under {root}")

    category_root = candidates[0]
    records: list[ImageRecord] = []
    for split in ("train", "test"):
        split_root = category_root / split
        if not split_root.exists():
            continue
        for image_path in list_images(split_root):
            relative = image_path.relative_to(split_root)
            parts = {part.lower() for part in relative.parts}
            is_good = "good" in parts or "normal" in parts
            defect_type = "good" if is_good else next(iter(parts - {"bad", "anomaly", "images"}), "defect")
            records.append(
                ImageRecord(
                    path=image_path,
                    label=0 if is_good else 1,
                    split=split,
                    category=category,
                    defect_type=defect_type,
                )
            )
    return records


def load_dataset(source: str, root: Path, category: str | None = None) -> list[ImageRecord]:
    if source == "simple":
        return load_simple_anomaly_dataset(root)
    if source == "mvtec":
        if not category:
            raise ValueError("--category is required for MVTec.")
        return load_mvtec_category(root, category)
    if source == "visa":
        if not category:
            raise ValueError("--category is required for VisA.")
        return load_visa_category(root, category)
    raise ValueError(f"Unsupported dataset source: {source}")


def write_records(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "path": str(record.path),
                    "label": record.label,
                    "split": record.split,
                    "category": record.category,
                    "defect_type": record.defect_type,
                }
                for record in records
            ],
            handle,
            indent=2,
        )
