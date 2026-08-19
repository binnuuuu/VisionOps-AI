# Training Pipeline

VisionOps now has three ML paths:

- **PatchCore-lite anomaly detection** for normal-sample learning and heatmaps.
- **Anomalib PatchCore/EfficientAD** for deep-feature anomaly detection through PyTorch.
- **YOLO object detection** for supervised blister/tablet/defect detection once labeled Roboflow or real-line data is available.

## PatchCore-Lite

Train on local synthetic blister data:

```bash
python scripts/train_anomaly.py \
  --source simple \
  --root data/demo_blisters \
  --run-name demo_patchcore_lite
```

More sensitive calibration for inspection workflows:

```bash
python scripts/train_anomaly.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --run-name visa_capsules_patchcore_lite_sensitive \
  --threshold-quantile 75 \
  --threshold-std-factor 0
```

Evaluate and generate heatmaps:

```bash
python scripts/evaluate_anomaly.py \
  --model data/training_runs/demo_patchcore_lite/patchcore_lite.npz \
  --source simple \
  --root data/demo_blisters \
  --heatmaps
```

Outputs:

- `data/training_runs/<run>/patchcore_lite.npz`
- `data/training_runs/<run>/train_summary.json`
- `data/training_runs/<run>/evaluation/eval_summary.json`
- `data/training_runs/<run>/evaluation/heatmaps/*.png`

For MVTec:

```bash
python scripts/train_anomaly.py \
  --source mvtec \
  --root datasets/raw/mvtec_ad \
  --category capsule \
  --run-name mvtec_capsule_patchcore_lite
```

For VisA:

```bash
python scripts/train_anomaly.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --run-name visa_capsules_patchcore_lite
```

## Anomalib

Prepare the VisA capsules Folder dataset:

```bash
python scripts/prepare_anomalib_dataset.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --out data/anomalib_datasets/visa_capsules \
  --mode symlink
```

Train the current CPU-safe PatchCore baseline:

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml run --rm \
  -e HF_HOME=/app/data/hf_cache \
  anomalib python scripts/train_anomalib.py \
    --dataset-root data/anomalib_datasets/visa_capsules \
    --run-name visa_capsules_anomalib_patchcore_resnet18 \
    --model patchcore \
    --backbone resnet18 \
    --layers layer2 layer3 \
    --coreset-sampling-ratio 0.02 \
    --image-size 192 \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --accelerator cpu \
    --devices 1
```

See `docs/anomalib_training.md` for full details and results.

## YOLO

After exporting a Roboflow dataset in YOLOv8 format:

```bash
python scripts/prepare_yolo_dataset.py \
  --root datasets/raw/roboflow_larger_blister_pack_defect
```

Install optional ML dependencies:

```bash
pip install -r backend/requirements-ml.txt
```

Train:

```bash
python scripts/train_yolo.py \
  --data configs/yolo/blister_defects.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640
```

## Notes

- PatchCore-lite is a local baseline that runs without PyTorch.
- Anomalib PatchCore is the current deep-feature baseline; GPU tuning should come next.
- YOLO should be trained only after class names and dataset license/export terms are confirmed.
