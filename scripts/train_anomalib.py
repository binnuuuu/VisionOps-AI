#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _make_pre_processor(image_size: int):
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms import v2

    return PreProcessor(
        transform=v2.Compose(
            [
                v2.Resize(size=(image_size, image_size), antialias=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    )


def _make_model(name: str, image_size: int, backbone: str, layers: list[str], coreset_sampling_ratio: float):
    from anomalib.models import EfficientAd, Patchcore

    if name == "patchcore":
        return Patchcore(
            backbone=backbone,
            layers=layers,
            pre_trained=True,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=9,
            pre_processor=_make_pre_processor(image_size),
        )
    if name == "efficientad":
        return EfficientAd(pre_processor=_make_pre_processor(image_size))
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep-feature Anomalib model.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "anomalib_datasets" / "visa_capsules")
    parser.add_argument("--run-name", default="visa_capsules_anomalib_patchcore")
    parser.add_argument("--model", choices=["patchcore", "efficientad"], default="patchcore")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "anomalib_runs")
    args = parser.parse_args()

    try:
        import anomalib
        import torch
        from anomalib.data import Folder
        from anomalib.engine import Engine
    except ImportError as exc:
        raise SystemExit(
            "Anomalib/PyTorch is not installed. Use the ML Docker image:\n"
            "  docker compose -f docker-compose.yml -f docker-compose.anomalib.yml build anomalib\n"
            "  docker compose -f docker-compose.yml -f docker-compose.anomalib.yml run --rm anomalib ...\n"
            "or install locally with: pip install 'anomalib[cpu]==2.5.0'"
        ) from exc

    run_dir = args.out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    datamodule = Folder(
        name="visa_capsules",
        root=args.dataset_root,
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        test_split_mode="from_dir",
        normal_split_ratio=0,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    model = _make_model(
        args.model,
        args.image_size,
        args.backbone,
        args.layers,
        args.coreset_sampling_ratio,
    )
    engine = Engine(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=run_dir,
    )

    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)

    checkpoints = sorted(run_dir.rglob("*.ckpt"))
    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "dataset_root": str(args.dataset_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anomalib_version": getattr(anomalib, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "image_size": args.image_size,
        "backbone": args.backbone if args.model == "patchcore" else None,
        "layers": args.layers if args.model == "patchcore" else None,
        "coreset_sampling_ratio": args.coreset_sampling_ratio if args.model == "patchcore" else None,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "test_results": _json_safe(test_results),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
