#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.visionops_ml.datasets import load_dataset, write_records
from ml.visionops_ml.patchcore_lite import PatchCoreLite, PatchCoreLiteConfig

DEFAULT_RUNS = ROOT / "data" / "training_runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local PatchCore-lite anomaly model.")
    parser.add_argument("--source", choices=["simple", "mvtec", "visa"], default="simple")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "demo_blisters")
    parser.add_argument("--category", help="Dataset category for MVTec/VisA, e.g. capsule or capsules.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-memory-patches", type=int, default=12000)
    parser.add_argument("--threshold-quantile", type=float, default=99.0)
    parser.add_argument("--threshold-std-factor", type=float, default=3.0)
    parser.add_argument("--min-threshold", type=float, default=0.01)
    args = parser.parse_args()

    records = load_dataset(args.source, args.root, args.category)
    normal_train = [record.path for record in records if record.split == "train" and record.label == 0]
    if not normal_train:
        raise SystemExit("No normal training images found.")

    run_name = args.run_name or f"{args.source}_{args.category or args.root.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out / run_name
    model_path = run_dir / "patchcore_lite.npz"
    config = PatchCoreLiteConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        patch_size=args.patch_size,
        stride=args.stride,
        max_memory_patches=args.max_memory_patches,
        threshold_quantile=args.threshold_quantile,
        threshold_std_factor=args.threshold_std_factor,
        min_threshold=args.min_threshold,
    )
    model = PatchCoreLite(config)
    fit_info = model.fit(normal_train)
    metadata = {
        "run_name": run_name,
        "source": args.source,
        "root": str(args.root),
        "category": args.category,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    model.save(model_path, metadata)
    write_records(records, run_dir / "dataset_records.json")

    summary = {
        **metadata,
        **fit_info,
        "model_path": str(model_path),
        "dataset_records": str(run_dir / "dataset_records.json"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
