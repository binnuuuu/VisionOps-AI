# Anomalib Deep-Feature Training

VisionOps now has a dedicated Anomalib/PyTorch path for production-grade anomaly baselines without bloating the normal FastAPI image.

## What Is Included

- `backend/Dockerfile.anomalib`: optional Python 3.11 CPU ML image.
- `docker-compose.anomalib.yml`: `anomalib` service with local `data`, `datasets`, `backend`, `ml`, and `scripts` mounted.
- `scripts/prepare_anomalib_dataset.py`: converts the VisA official split into Anomalib Folder layout.
- `scripts/train_anomalib.py`: trains PatchCore or EfficientAD through Anomalib.
- `data/sample_images/visa_capsules`: 6 normal and 6 anomaly sample images copied into the project for quick manual tests.

## Dataset Prep

The current real public baseline uses the VisA `capsules` category:

```bash
python scripts/prepare_anomalib_dataset.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --out data/anomalib_datasets/visa_capsules \
  --mode symlink
```

Prepared counts:

- Train good: 542
- Test good: 60
- Test anomaly: 100

## Docker Build

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml build anomalib
```

The Anomalib image uses `anomalib[cpu]==2.5.0`, PyTorch CPU, torchvision, timm, Lightning, and headless OpenCV.

## PatchCore Training

This CPU-safe profile completed in Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml run --rm \
  -e HF_HOME=/app/data/hf_cache \
  -e HF_HUB_OFFLINE=1 \
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
    --num-workers 0 \
    --max-epochs 1 \
    --accelerator cpu \
    --devices 1
```

Result:

- Image AUROC: 0.8140
- Image F1: 0.78125
- Summary: `data/anomalib_runs/visa_capsules_anomalib_patchcore_resnet18/summary.json`
- Checkpoint: `data/anomalib_runs/visa_capsules_anomalib_patchcore_resnet18/Patchcore/visa_capsules/v0/weights/lightning/model.ckpt`

## EfficientAD

EfficientAD is wired through the same script:

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml run --rm anomalib \
  python scripts/train_anomalib.py \
    --dataset-root data/anomalib_datasets/visa_capsules \
    --run-name visa_capsules_anomalib_efficientad \
    --model efficientad \
    --image-size 192 \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --accelerator cpu \
    --devices 1
```

For a serious EfficientAD run, use GPU and allow enough epochs. The model initializes correctly in the current container, but the completed trained baseline for this step is PatchCore.

## Notes

- Docker networking was intermittent during the run, so ResNet-18 weights were seeded into `data/hf_cache` from the host and training was run with `HF_HUB_OFFLINE=1`.
- Generated caches and run outputs are ignored by git; the sample image pack is intentionally kept in the project.
- The current checkpoint is a baseline for platform development, not a validated pharmaceutical inspection model.
