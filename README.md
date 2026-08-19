# VisionOps Pharma Inspection

VisionOps is an MVP platform inspired by SPINE ULTRA-style AI visual inspection machines for tablets, capsules, softgels, and blister packs. It gives you a working software loop now: create a machine recipe, teach it with good samples, inspect single images or batches, view localized heatmaps, classify defect types, simulate active sorting, approve model versions, and keep a QA audit trail.

## What Is Built

- FastAPI backend for product recipes, teaching, inspection, dataset catalog, image storage, and inspection history.
- React HMI dashboard for operator-style workflows inspired by the uploaded PDF.
- Product recipes for round tablets, oblong tablets, capsules, softgels, and blister packs.
- Recipe metadata for shape constraints, dimensions, six-side inspection, colour/backlight/3D optical channels, and sorting mode.
- Lightweight normal-sample anomaly model that works without real defect data.
- Defect localization with labeled heatmap boxes, severity, confidence, and cavity-level area estimates.
- Batch inspection endpoint and dashboard report with good/review/reject counts, reject rate, and defect type counts.
- Live Three.js digital twin for the feeder, camera, inspection station, AI unit, conveyor, reject gate, sorting bins, and operator panel.
- Backend-owned twin state streamed through WebSockets, with sequential batch playback, reject routing, actuator animation, bin fill/count updates, event timeline, and inspection evidence.
- Model version registry with approve and activate actions for trained product recipes.
- Runtime performance telemetry for duration percentiles, inference time, and model cache hit rate.
- Durable audit events for recipe creation, model training, model approval, activation, single inspections, and batch runs.
- Synthetic blister image generator for demos and tests.
- Dataset catalog and sourcing script for public anomaly/blister datasets.
- PatchCore-lite training/evaluation pipeline for normal-sample anomaly detection.
- Deep-feature Anomalib training path for PatchCore and EfficientAD in a dedicated PyTorch Docker image.
- YOLO dataset validation and training wrapper for supervised blister defect detection.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python scripts/make_synthetic_blisters.py
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

If `pnpm` is not installed, `npm install` and `npm run dev` also work.

Open the web port configured by `VISIONOPS_WEB_PORT` in `.env` (the current workspace uses `http://127.0.0.1:15173`).

## Docker Start

```bash
cp .env.docker.example .env
docker compose up --build
```

Open `http://127.0.0.1:5173`.

See `docs/docker.md` for Docker training, dataset, and test commands.

## Demo Flow

1. Use the ready Capsule Demo recipe or create a new tablet/capsule/softgel recipe.
2. Teach it using 3 or more good product images.
3. Inspect one image from `data/sample_images/visa_capsules/anomaly` or `data/demo_blisters/defect`.
4. Run a batch with several images from the same sample folder.
5. Review the heatmap, score, defect type, severity, confidence, batch QA report, model version, audit events, and latency telemetry.

## Digital Twin

The twin is an operating view of backend inspection state, not a separate 3D demo. Its typed model includes the plant, inspection line, assets, recipe, batch, active model, inspection counters, route state, and inspection events.

- Snapshot: `GET /api/twin/state`
- Live stream: `WS /api/twin/ws`
- Source model: `backend/app/twin.py`
- Frontend view: `frontend/src/DigitalTwin.tsx`

Every real single or batch inspection updates the twin. Batch images enter an ordered runtime queue and move through feeding, camera capture, inspection, AI decision, and sorting phases. The operator panel shows the current source frame, switches to localized evidence when the decision is available, and tracks each image in the batch. Reject results raise the camera and inspection warning states, animate the product route and reject gate, increment the reject bin, and add a timeline event. The latest recipe and evidence are restored from persisted inspection records after a backend restart.

## Dataset Sourcing

See `docs/dataset_sourcing.md` for the dataset rationale.
See `docs/training_pipeline.md` and `docs/anomalib_training.md` for training commands.

List curated datasets:

```bash
python scripts/source_datasets.py list
```

Download VisA, which is directly available under CC BY 4.0:

```bash
python scripts/source_datasets.py download visa --extract
```

MVTec AD is excellent for anomaly baselines, but its license is CC BY-NC-SA 4.0 and the official page requires license acceptance. Download it from the official page, then register/extract it:

```bash
python scripts/source_datasets.py download mvtec_ad --archive /path/to/mvtec_anomaly_detection.tar.xz --extract
```

Roboflow blister datasets require an API key for export:

```bash
export ROBOFLOW_API_KEY=...
python scripts/source_datasets.py download roboflow_larger_blister_pack_defect --extract
```

## Why This Architecture

The PDF describes a machine that learns from a small set of good blisters. That is closer to anomaly detection than a purely supervised detector, especially when you do not yet have real defect data. This MVP starts with normal-only inspection and leaves a clear path to add PatchCore, PaDiM, EfficientAD, and YOLO training once public and real data are available.

## Next Engineering Steps

- Tune Anomalib PatchCore/EfficientAD on GPU and later on real blister-line images.
- Train YOLO on blister-specific Roboflow datasets.
- Add camera/live stream ingestion.
- Add reject-signal simulation, then real PLC/GPIO integration.
- Add role-based access, electronic signatures, and CFR 21 Part 11 style controls.
