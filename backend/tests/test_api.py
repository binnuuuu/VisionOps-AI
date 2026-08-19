from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, store
from scripts.make_synthetic_blisters import blister_image


client = TestClient(app)


@pytest.fixture(autouse=True)
def restore_store_after_test():
    snapshot = deepcopy(store.read())
    yield

    def mutator(state: dict) -> None:
        state.clear()
        state.update(deepcopy(snapshot))

    store.update(mutator)


def _image(path: Path, seed: int, defect: str | None = None) -> Path:
    blister_image(seed, defect, 4, 3).save(path)
    return path


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_digital_twin_state_and_websocket() -> None:
    response = client.get("/api/twin/state")
    assert response.status_code == 200
    body = response.json()
    assert body["plant"]["plant_id"] == "PLANT_01"
    assert body["batch"]["batch_id"]
    assert any(asset["asset_id"] == "CAMERA_01" and asset["status"] in {"online", "alert"} for asset in body["assets"])
    assert any(asset["asset_id"] == "BIN_REJECT_01" and asset["metrics"]["fill_pct"] >= 63 for asset in body["assets"])

    with client.websocket_connect("/api/twin/ws") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["connection_state"] == "live"
        assert snapshot["inspection_line"]["status"] == "running"


def test_dashboard_assistant_answers_from_live_state() -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "Are the camera and reject gate online?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "asset_status"
    assert "camera" in body["answer"].lower()
    assert "reject gate" in body["answer"].lower()
    assert body["context"]["twin_sequence"] >= 1
    assert len(body["suggested_questions"]) >= 3


def test_dashboard_assistant_reports_decision_counts() -> None:
    response = client.post(
        "/api/assistant/ask",
        json={"question": "How many products were accepted and rejected?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "decision_counts"
    assert "accepted" in body["answer"].lower()
    assert "rejected" in body["answer"].lower()


def test_training_runs_endpoint() -> None:
    response = client.get("/api/training/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_product_teach_and_inspect_flow(tmp_path: Path) -> None:
    create_response = client.post(
        "/api/products",
        data={"name": "Test blister", "sku": "T-001", "cavity_rows": "4", "cavity_cols": "3"},
    )
    assert create_response.status_code == 200
    product_id = create_response.json()["product"]["id"]

    files = []
    handles = []
    try:
        for index in range(5):
            path = _image(tmp_path / f"good_{index}.png", 600 + index)
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "image/png")))
        teach_response = client.post(f"/api/products/{product_id}/teach", files=files)
    finally:
        for handle in handles:
            handle.close()

    assert teach_response.status_code == 200
    taught_product = teach_response.json()["product"]
    assert taught_product["status"] == "ready"
    assert taught_product["active_model_version_id"]

    versions_response = client.get("/api/model-versions")
    assert versions_response.status_code == 200
    assert any(version["id"] == taught_product["active_model_version_id"] for version in versions_response.json())

    approve_response = client.post(
        f"/api/model-versions/{taught_product['active_model_version_id']}/approve",
        data={"approved_by": "pytest"},
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["model_version"]["status"] == "approved"

    activate_response = client.post(f"/api/model-versions/{taught_product['active_model_version_id']}/activate")
    assert activate_response.status_code == 200
    assert activate_response.json()["model_version"]["status"] == "active"

    defect_path = _image(tmp_path / "defect.png", 900, "spot")
    with defect_path.open("rb") as handle:
        inspect_response = client.post(
            "/api/inspect/upload",
            data={"product_id": product_id},
            files={"file": (defect_path.name, handle, "image/png")},
        )
    assert inspect_response.status_code == 200
    body = inspect_response.json()["inspection"]
    assert body["decision"] in {"pass", "review", "reject"}
    assert body["heatmap_url"].startswith("/media/")
    assert body["model_version_id"] == taught_product["active_model_version_id"]
    assert "inference" in body["timings_ms"]
    if body["defect_regions"]:
        assert body["defect_regions"][0]["defect_type"]
        assert body["defect_regions"][0]["severity"] in {"moderate", "high", "critical"}

    batch_files = []
    batch_handles = []
    try:
        for index, defect in enumerate(("missing", "spot")):
            path = _image(tmp_path / f"batch_{index}.png", 950 + index, defect)
            handle = path.open("rb")
            batch_handles.append(handle)
            batch_files.append(("files", (path.name, handle, "image/png")))
        batch_response = client.post(
            "/api/inspect/batch",
            data={"product_id": product_id, "batch_id": "pytest-batch"},
            files=batch_files,
        )
    finally:
        for handle in batch_handles:
            handle.close()

    assert batch_response.status_code == 200
    assert len(batch_response.json()) == 2
    assert all(item["batch_id"] == "pytest-batch" for item in batch_response.json())

    report_response = client.get("/api/reports/batch", params={"product_id": product_id})
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["total_inspections"] >= 3
    assert report["pass_count"] + report["review_count"] + report["reject_count"] == report["total_inspections"]

    performance_response = client.get("/api/performance")
    assert performance_response.status_code == 200
    assert performance_response.json()["total_inspections"] >= 3

    audit_response = client.get("/api/audit/events")
    assert audit_response.status_code == 200
    assert any(event["event_type"] == "batch.completed" for event in audit_response.json())
