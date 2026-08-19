#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small included image pack for demo inference.")
    parser.add_argument("--visa-root", type=Path, default=ROOT / "datasets" / "raw" / "visa")
    parser.add_argument("--category", default="capsules")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sample_images" / "visa_capsules")
    parser.add_argument("--normal", type=int, default=6)
    parser.add_argument("--anomaly", type=int, default=6)
    args = parser.parse_args()

    split_csv = args.visa_root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing VisA split CSV: {split_csv}")

    buckets = {"normal": [], "anomaly": []}
    with split_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("object") != args.category or row.get("split") != "test":
                continue
            label = "normal" if row.get("label") == "normal" else "anomaly"
            buckets[label].append(args.visa_root / row["image"])

    selected = {
        "normal": buckets["normal"][: args.normal],
        "anomaly": buckets["anomaly"][: args.anomaly],
    }

    records = []
    for label, paths in selected.items():
        target_dir = args.out / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dest = target_dir / f"{args.category}_{label}_{src.name.lower()}"
            shutil.copy2(src, dest)
            records.append({"label": label, "path": str(dest.relative_to(args.out)), "source": str(src)})

    manifest = {
        "name": "VisA capsules sample image pack",
        "source_dataset": "VisA",
        "source_url": "https://registry.opendata.aws/visa/",
        "license": "CC BY 4.0",
        "category": args.category,
        "counts": {label: len(paths) for label, paths in selected.items()},
        "records": records,
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

