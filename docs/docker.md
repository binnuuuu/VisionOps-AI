# Docker Setup

VisionOps runs as two Docker services:

- `backend`: FastAPI, inspection API, local model/training scripts.
- `frontend`: nginx-served React dashboard, with `/api` and `/media` proxied to `backend`.

The compose stack mounts local folders into the backend container:

- `./data:/app/data`
- `./datasets:/app/datasets`

This keeps datasets, training runs, inspection images, and model artifacts on your machine instead of baking them into images.

## Start The App

```bash
cp .env.docker.example .env
docker compose up --build
```

Open:

- Dashboard: http://127.0.0.1:5173
- API: http://127.0.0.1:8000

If a port is busy, edit `.env`:

```bash
VISIONOPS_WEB_PORT=5174
VISIONOPS_API_PORT=8001
```

## Run Backend Tests In Docker

```bash
docker compose run --rm backend python -m pytest -q
```

## Generate Demo Blisters

```bash
docker compose run --rm backend python scripts/make_synthetic_blisters.py
```

## Train PatchCore-Lite In Docker

Synthetic demo:

```bash
docker compose run --rm backend \
  python scripts/train_anomaly.py \
  --source simple \
  --root data/demo_blisters \
  --run-name demo_patchcore_lite
```

VisA capsules, after the dataset is downloaded/extracted:

```bash
docker compose run --rm backend \
  python scripts/train_anomaly.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --run-name visa_capsules_patchcore_lite_hires_sensitive \
  --image-width 512 \
  --image-height 384 \
  --patch-size 32 \
  --stride 16 \
  --max-memory-patches 24000 \
  --threshold-quantile 75 \
  --threshold-std-factor 0
```

Evaluate:

```bash
docker compose run --rm backend \
  python scripts/evaluate_anomaly.py \
  --model data/training_runs/visa_capsules_patchcore_lite_hires_sensitive/patchcore_lite.npz \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --heatmaps
```

## Train Anomalib PatchCore In Docker

Build the optional PyTorch/Anomalib image:

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml build anomalib
```

Prepare the VisA capsules split for Anomalib's Folder datamodule:

```bash
docker compose -f docker-compose.yml -f docker-compose.anomalib.yml run --rm anomalib \
  python scripts/prepare_anomalib_dataset.py \
  --source visa \
  --root datasets/raw/visa \
  --category capsules \
  --out data/anomalib_datasets/visa_capsules \
  --mode symlink
```

Run the memory-safe CPU PatchCore baseline:

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
    --num-workers 0 \
    --accelerator cpu \
    --devices 1
```

Outputs are written to `data/anomalib_runs/<run>/`.
If the pretrained weights are already cached and Docker networking is unavailable, add `-e HF_HUB_OFFLINE=1`.

## Download Datasets In Docker

```bash
docker compose run --rm backend python scripts/source_datasets.py list
docker compose run --rm backend python scripts/source_datasets.py download visa --extract
```

Roboflow exports require `ROBOFLOW_API_KEY` in `.env`.

## Stop And Clean

Stop containers:

```bash
docker compose down
```

Remove built images:

```bash
docker compose down --rmi local
```

This does not delete `./data` or `./datasets`.
