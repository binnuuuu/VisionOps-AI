#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO blister defect detector with Ultralytics.")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "yolo" / "blister_defects.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "data" / "yolo_runs")
    parser.add_argument("--name", default="blister_defects")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install the optional ML stack first:\n"
            "  pip install -r backend/requirements-ml.txt\n"
            "For GPU training, install the PyTorch build recommended for your hardware before running this script."
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    print(results)


if __name__ == "__main__":
    main()

