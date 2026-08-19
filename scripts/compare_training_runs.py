#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VisionOps training runs.")
    parser.add_argument("--runs", type=Path, default=ROOT / "data" / "training_runs")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.runs.iterdir()):
        train_path = run_dir / "train_summary.json"
        eval_path = run_dir / "evaluation" / "eval_summary.json"
        if not train_path.exists():
            continue
        train = json.loads(train_path.read_text())
        metrics = {}
        if eval_path.exists():
            metrics = json.loads(eval_path.read_text()).get("metrics", {})
        rows.append(
            {
                "run": train.get("run_name", run_dir.name),
                "category": train.get("category"),
                "threshold": metrics.get("threshold", train.get("threshold")),
                "auroc": metrics.get("auroc"),
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "tp": metrics.get("tp"),
                "tn": metrics.get("tn"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
            }
        )

    if not rows:
        print("No training runs found.")
        return

    headers = ["run", "threshold", "auroc", "f1", "precision", "recall", "specificity", "tp", "tn", "fp", "fn"]
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


if __name__ == "__main__":
    main()

