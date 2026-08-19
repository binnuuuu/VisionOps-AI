from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("VISIONOPS_DATA_DIR", REPO_ROOT / "data")).resolve()
MEDIA_DIR = DATA_DIR / "media"
MODELS_DIR = DATA_DIR / "models"
TRAINING_RUNS_DIR = DATA_DIR / "training_runs"
ANOMALIB_RUNS_DIR = DATA_DIR / "anomalib_runs"
STORE_FILE = DATA_DIR / "visionops_store.json"
DATASET_CATALOG = REPO_ROOT / "datasets" / "catalog.json"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, MEDIA_DIR, MODELS_DIR, TRAINING_RUNS_DIR, ANOMALIB_RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
