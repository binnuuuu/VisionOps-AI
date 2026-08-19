#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "hardlink":
        os.link(src, dest)
    else:
        relative_src = os.path.relpath(src, start=dest.parent)
        dest.symlink_to(relative_src)


def prepare_visa_capsules(root: Path, category: str, out: Path, mode: str) -> dict:
    split_csv = root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing VisA split CSV: {split_csv}")

    counts = {"train/good": 0, "test/good": 0, "test/anomaly": 0}
    records = []
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("object") != category:
                continue
            label = row.get("label", "").lower()
            split = row.get("split", "test")
            src = root / row["image"]
            if not src.exists():
                raise FileNotFoundError(src)

            if split == "train" and label == "normal":
                dest_dir = out / "train" / "good"
                key = "train/good"
            elif split == "test" and label == "normal":
                dest_dir = out / "test" / "good"
                key = "test/good"
            elif split == "test":
                dest_dir = out / "test" / "anomaly"
                key = "test/anomaly"
            else:
                continue

            dest = dest_dir / src.name
            link_or_copy(src, dest, mode)
            counts[key] += 1
            records.append(
                {
                    "source": str(src),
                    "target": str(dest),
                    "split": split,
                    "label": label,
                }
            )

    manifest = {
        "name": f"visa_{category}_anomalib",
        "source_dataset": "VisA",
        "source_url": "https://registry.opendata.aws/visa/",
        "license": "CC BY 4.0",
        "category": category,
        "layout": "anomalib Folder",
        "mode": mode,
        "counts": counts,
        "records": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for Anomalib Folder datamodule.")
    parser.add_argument("--source", choices=["visa"], default="visa")
    parser.add_argument("--root", type=Path, default=ROOT / "datasets" / "raw" / "visa")
    parser.add_argument("--category", default="capsules")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "anomalib_datasets" / "visa_capsules")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    args = parser.parse_args()

    if args.source != "visa":
        raise SystemExit(f"Unsupported source: {args.source}")

    manifest = prepare_visa_capsules(args.root.resolve(), args.category, args.out.resolve(), args.mode)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

