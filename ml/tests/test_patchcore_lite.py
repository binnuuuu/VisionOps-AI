from __future__ import annotations

from pathlib import Path

from ml.visionops_ml.datasets import load_simple_anomaly_dataset
from ml.visionops_ml.metrics import binary_metrics
from ml.visionops_ml.patchcore_lite import PatchCoreLite, PatchCoreLiteConfig
from scripts.make_synthetic_blisters import blister_image


def test_patchcore_lite_training_and_prediction(tmp_path: Path) -> None:
    good_dir = tmp_path / "good"
    defect_dir = tmp_path / "defect"
    good_dir.mkdir()
    defect_dir.mkdir()
    for index in range(10):
        blister_image(1200 + index, None, 4, 3).save(good_dir / f"good_{index}.png")
    blister_image(3000, "missing", 4, 3).save(defect_dir / "missing_001.png")
    blister_image(3001, "spot", 4, 3).save(defect_dir / "spot_001.png")

    records = load_simple_anomaly_dataset(tmp_path)
    train_paths = [record.path for record in records if record.split == "train" and record.label == 0]
    test_records = [record for record in records if record.split == "test" or record.label == 1]

    model = PatchCoreLite(PatchCoreLiteConfig(image_width=192, image_height=128, max_memory_patches=1000))
    fit_info = model.fit(train_paths)
    assert fit_info["normal_images"] >= 3
    assert fit_info["memory_patches"] > 0

    labels = []
    scores = []
    for record in test_records:
        prediction = model.predict(record.path)
        labels.append(record.label)
        scores.append(prediction.score)

    metrics = binary_metrics(labels, scores, model.threshold or 0.0)
    assert metrics["count"] == len(test_records)
    assert max(scores) > 0

