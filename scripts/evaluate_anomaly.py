#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.visionops_ml.datasets import load_dataset
from ml.visionops_ml.metrics import auroc, binary_metrics
from ml.visionops_ml.patchcore_lite import PatchCoreLite


def safe_heatmap_name(root: Path, image_path: Path) -> str:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        relative = image_path.name
    if not isinstance(relative, Path):
        relative = Path(str(relative))
    stem = "_".join(relative.with_suffix("").parts)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem) + "_heatmap.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PatchCore-lite anomaly model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", choices=["simple", "mvtec", "visa"], default="simple")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "demo_blisters")
    parser.add_argument("--category", help="Dataset category for MVTec/VisA.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Override model threshold for metrics/predictions.")
    args = parser.parse_args()

    records = load_dataset(args.source, args.root, args.category)
    evaluation_records = [
        record
        for record in records
        if record.split == "test" or record.label == 1
    ]
    if not evaluation_records:
        raise SystemExit("No evaluation images found.")

    model = PatchCoreLite.load(args.model)
    operating_threshold = args.threshold if args.threshold is not None else model.threshold
    out_dir = args.out or args.model.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels: list[int] = []
    scores: list[float] = []
    for record in evaluation_records:
        prediction = model.predict(record.path)
        is_anomaly = prediction.score >= float(operating_threshold)
        labels.append(record.label)
        scores.append(prediction.score)
        heatmap_path = None
        if args.heatmaps:
            heatmap_path = out_dir / "heatmaps" / safe_heatmap_name(args.root, record.path)
            model.heatmap(record.path, prediction, heatmap_path)
        rows.append(
            {
                "path": str(record.path),
                "label": record.label,
                "defect_type": record.defect_type,
                "score": prediction.score,
                "threshold": operating_threshold,
                "prediction": int(is_anomaly),
                "heatmap": str(heatmap_path) if heatmap_path else None,
            }
        )

    metrics = binary_metrics(labels, scores, float(operating_threshold))
    metrics["auroc"] = auroc(labels, scores)
    result = {"metrics": metrics, "predictions": rows}
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
