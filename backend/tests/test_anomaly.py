from pathlib import Path

from backend.app.anomaly import inspect_with_model, inspect_with_patchcore_lite_model, train_normal_model
from scripts.make_synthetic_blisters import blister_image


def test_normal_model_flags_synthetic_defect(tmp_path: Path) -> None:
    good_paths = []
    for index in range(12):
        path = tmp_path / f"good_{index}.png"
        blister_image(100 + index, None, 4, 3).save(path)
        good_paths.append(path)

    model_path = tmp_path / "model.npz"
    train_info = train_normal_model(good_paths, model_path, rows=4, cols=3)
    assert model_path.exists()
    assert train_info["sample_count"] == 12
    assert train_info["threshold"] >= 4.6

    normal_path = tmp_path / "normal_holdout.png"
    blister_image(500, None, 4, 3).save(normal_path)
    normal_result = inspect_with_model(model_path, normal_path)
    assert normal_result.decision == "pass"
    assert normal_result.defect_regions == []

    defect_path = tmp_path / "missing.png"
    blister_image(400, "missing", 4, 3).save(defect_path)
    result = inspect_with_model(model_path, defect_path)

    assert result.score > 0
    assert result.threshold > 0
    assert result.decision == "reject"
    assert result.defect_regions
    assert result.timings_ms["inference"] >= 0
    assert result.timings_ms["heatmap"] >= 0
    region = result.defect_regions[0]
    assert region["defect_type"]
    assert region["defect_type"] != "visual_anomaly"
    assert region["severity"] in {"moderate", "high", "critical"}
    assert 0 <= region["confidence"] <= 1
    assert region["area_pct"] >= 0

    cached_result = inspect_with_model(model_path, defect_path)
    assert cached_result.model_cache_hit is True


def test_patchcore_rejects_partial_ring_bubble() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "data" / "training_runs" / "visa_capsules_patchcore_lite_p16" / "patchcore_lite.npz"
    bubble_path = root / "data" / "sample_images" / "visa_capsules" / "anomaly" / "capsules_anomaly_004.jpg"

    result = inspect_with_patchcore_lite_model(model_path, bubble_path)

    assert result.decision == "reject"
    assert result.defect_regions
    assert result.defect_regions[0]["defect_type"] == "bubble_or_fill_void"
