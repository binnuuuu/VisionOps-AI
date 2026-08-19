#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "datasets" / "catalog.json"
DEFAULT_OUT = ROOT / "datasets" / "raw"


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["datasets"]


def find_dataset(dataset_id: str) -> dict:
    for dataset in load_catalog():
        if dataset["id"] == dataset_id:
            return dataset
    raise SystemExit(f"Unknown dataset id: {dataset_id}")


def print_catalog() -> None:
    for dataset in sorted(load_catalog(), key=lambda item: item["priority"]):
        print(f"{dataset['priority']}. {dataset['id']} - {dataset['name']}")
        print(f"   task: {dataset['task']}")
        print(f"   fit: {dataset['domain_fit']}")
        print(f"   license: {dataset['license']}")
        print(f"   source: {dataset['source_url']}")
        print()


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"Saving to {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if archive.name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
        return
    raise SystemExit(f"Unsupported archive type: {archive}")


def download_visa(out_dir: Path, extract: bool) -> None:
    dataset = find_dataset("visa")
    archive = out_dir / "VisA_20220922.tar"
    if not archive.exists():
        download_file(dataset["download_url"], archive)
    else:
        print(f"Archive already exists: {archive}")
    if extract:
        extract_archive(archive, out_dir / "visa")


def prepare_mvtec(out_dir: Path, archive: Path | None, extract: bool) -> None:
    dataset = find_dataset("mvtec_ad")
    if archive is None:
        print("MVTec AD requires manual license acceptance before download.")
        print(f"Open: {dataset['source_url']}")
        print("After downloading, run:")
        print("  python scripts/source_datasets.py download mvtec_ad --archive /path/to/mvtec_anomaly_detection.tar.xz --extract")
        raise SystemExit(2)
    if not archive.exists():
        raise SystemExit(f"Archive does not exist: {archive}")
    target_archive = out_dir / archive.name
    if archive.resolve() != target_archive.resolve():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, target_archive)
    print(f"Registered MVTec archive: {target_archive}")
    if extract:
        extract_archive(target_archive, out_dir / "mvtec_ad")


def download_roboflow(dataset_id: str, out_dir: Path, extract: bool) -> None:
    dataset = find_dataset(dataset_id)
    token = os.getenv("ROBOFLOW_API_KEY")
    if not token:
        print("This Roboflow dataset requires ROBOFLOW_API_KEY.")
        print(f"Open: {dataset['source_url']}")
        print("Then set ROBOFLOW_API_KEY and rerun this command.")
        raise SystemExit(2)
    info = dataset["roboflow"]
    url = (
        f"https://universe.roboflow.com/{info['workspace']}/{info['project']}"
        f"/dataset/{info['version']}/download/{info['format']}?key={token}"
    )
    archive = out_dir / f"{dataset_id}.zip"
    download_file(url, archive)
    if extract:
        extract_archive(archive, out_dir / dataset_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Source public datasets for VisionOps.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List curated dataset sources.")

    download = sub.add_parser("download", help="Download or register a dataset.")
    download.add_argument("dataset_id", help="Dataset id from scripts/source_datasets.py list")
    download.add_argument("--out", type=Path, default=DEFAULT_OUT)
    download.add_argument("--archive", type=Path, help="Existing archive for manual-license datasets.")
    download.add_argument("--extract", action="store_true", help="Extract after download/registration.")

    args = parser.parse_args()
    if args.command == "list":
        print_catalog()
        return

    dataset_id = args.dataset_id
    out_dir = args.out.resolve()
    if dataset_id == "visa":
        download_visa(out_dir, args.extract)
    elif dataset_id == "mvtec_ad":
        prepare_mvtec(out_dir, args.archive, args.extract)
    elif dataset_id.startswith("roboflow_"):
        download_roboflow(dataset_id, out_dir, args.extract)
    else:
        raise SystemExit(f"No downloader implemented for {dataset_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

