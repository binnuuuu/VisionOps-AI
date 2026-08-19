from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .anomaly import inspect_with_model, inspect_with_patchcore_lite_model, train_normal_model
from .assistant import answer_dashboard_question
from .config import ANOMALIB_RUNS_DIR, DATASET_CATALOG, MEDIA_DIR, MODELS_DIR, SAMPLE_IMAGES_DIR, TRAINING_RUNS_DIR, ensure_runtime_dirs
from .schemas import (
    AssistantAnswer,
    AssistantQuestion,
    HealthResponse,
    Inspection,
    InspectionResponse,
    AuditEvent,
    BatchReport,
    ModelVersion,
    ModelVersionActionResponse,
    PerformanceStats,
    Product,
    ProductCreateResponse,
    TeachResponse,
    TrainingRun,
    TwinSnapshot,
)
from .storage import JsonStore
from .twin import DigitalTwin


ensure_runtime_dirs()
store = JsonStore()
twin = DigitalTwin()
twin.hydrate(store.read())

app = FastAPI(
    title="VisionOps Pharma Inspection API",
    version="0.1.0",
    description="MVP API for AI-assisted blister pack inspection workflows.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
if SAMPLE_IMAGES_DIR.exists():
    app.mount("/sample-images", StaticFiles(directory=str(SAMPLE_IMAGES_DIR)), name="sample-images")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_name(filename: str | None) -> str:
    base = Path(filename or "upload.png").name
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in "._-")
    return cleaned or "upload.png"


def _channels_from_form(value: str | None) -> list[str]:
    fallback = ["colour", "backlight", "3d"]
    if not value:
        return fallback
    channels = []
    for item in value.split(","):
        channel = item.strip().lower()
        if channel and channel not in channels:
            channels.append(channel)
    return channels or fallback


def _default_shape(product_family: str) -> str:
    return {
        "round_tablet": "round",
        "oblong_tablet": "oval_oblong",
        "capsule": "capsule",
        "softgel": "softgel",
        "blister_pack": "blister_grid",
    }.get(product_family, "capsule")


def _default_inspection_mode(product_family: str) -> str:
    return "blister_pack" if product_family == "blister_pack" else "loose_product"


def _default_inspection_sides(product_family: str, requested: int | None) -> int:
    if requested:
        return requested
    return 1 if product_family == "blister_pack" else 6


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return dest


def _looks_like_loose_capsule_image(path: Path) -> bool:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return False

    fitted = ImageOps.fit(image, (384, 256), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    arr = np.asarray(fitted, dtype=np.float32) / 255.0
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    maximum = np.maximum.reduce([red, green, blue])
    minimum = np.minimum.reduce([red, green, blue])
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    gray = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    capsule_pixels = (
        (green > red * 1.04)
        & (green > blue * 1.10)
        & (saturation > 0.20)
        & (gray > 0.22)
        & (gray < 0.92)
    )
    return bool(np.mean(capsule_pixels) > 0.045)


def _to_media_url(path: Path) -> str:
    return "/" + path.resolve().relative_to(MEDIA_DIR.parent.resolve()).as_posix()


def _to_sample_url(path: Path) -> str:
    return "/sample-images/" + path.resolve().relative_to(SAMPLE_IMAGES_DIR.resolve()).as_posix()


def _sample_pack_dir(pack_id: str) -> Path:
    safe_pack_id = "".join(ch for ch in pack_id if ch.isalnum() or ch in "._-")
    pack_dir = SAMPLE_IMAGES_DIR / safe_pack_id
    if not safe_pack_id or not pack_dir.exists() or not pack_dir.is_dir():
        raise HTTPException(status_code=404, detail="Sample image pack not found.")
    return pack_dir


def _load_sample_pack(pack_id: str = "visa_capsules") -> dict:
    pack_dir = _sample_pack_dir(pack_id)
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Sample image manifest not found.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    records = []
    for index, record in enumerate(manifest.get("records", []), start=1):
        relative_path = Path(str(record.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        image_path = pack_dir / relative_path
        if not image_path.exists():
            continue
        label = str(record.get("label") or "unknown")
        sample_id = f"{pack_id}:{relative_path.as_posix()}"
        records.append(
            {
                "id": sample_id,
                "pack_id": pack_id,
                "label": label,
                "display_name": image_path.stem.replace("_", " ").replace("capsules ", "").title(),
                "defect_type": record.get("defect_type"),
                "source": record.get("source"),
                "image_url": _to_sample_url(image_path),
                "index": index,
            }
        )

    return {
        "pack_id": pack_id,
        "name": manifest.get("name", pack_id.replace("_", " ").title()),
        "category": manifest.get("category", "capsules"),
        "source_dataset": manifest.get("source_dataset"),
        "license": manifest.get("license"),
        "counts": manifest.get("counts", {}),
        "records": records,
    }


def _resolve_sample_record(sample_id: str) -> tuple[dict, Path]:
    if ":" not in sample_id:
        raise HTTPException(status_code=400, detail=f"Invalid sample id: {sample_id}")
    pack_id, relative = sample_id.split(":", 1)
    pack = _load_sample_pack(pack_id)
    record = next((item for item in pack["records"] if item["id"] == sample_id), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")
    image_path = _sample_pack_dir(pack_id) / relative
    return record, image_path


def _resolve_model_path(stored_path: str) -> Path:
    path = Path(stored_path)
    if path.exists():
        return path
    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        candidate = MODELS_DIR.parent.joinpath(*parts[data_index + 1 :])
        if candidate.exists():
            return candidate
    if "models" in parts:
        model_index = parts.index("models")
        candidate = MODELS_DIR.joinpath(*parts[model_index + 1 :])
        if candidate.exists():
            return candidate
    return path


def _find_product(state: dict, product_id: str) -> dict:
    for product in state["products"]:
        if product["id"] == product_id:
            return product
    raise HTTPException(status_code=404, detail="Product not found")


def _audit_event(
    event_type: str,
    message: str,
    *,
    product_id: str | None = None,
    inspection_id: str | None = None,
    model_version_id: str | None = None,
    details: dict | None = None,
) -> dict:
    return AuditEvent(
        id=str(uuid.uuid4()),
        event_type=event_type,
        message=message,
        product_id=product_id,
        inspection_id=inspection_id,
        model_version_id=model_version_id,
        details=details or {},
        created_at=_now(),
    ).model_dump(mode="json")


def _append_audit(state: dict, event: dict) -> None:
    state.setdefault("audit_events", []).insert(0, event)
    state["audit_events"] = state["audit_events"][:500]


def _find_model_version(state: dict, version_id: str) -> dict:
    for version in state.get("model_versions", []):
        if version["id"] == version_id:
            return version
    raise HTTPException(status_code=404, detail="Model version not found")


def _next_product_model_version(state: dict, product_id: str) -> int:
    versions = [
        int(version.get("version", 0))
        for version in state.get("model_versions", [])
        if version.get("product_id") == product_id
    ]
    return (max(versions) if versions else 0) + 1


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=np.float32), percentile)), 3)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = store.read()
    return HealthResponse(
        ok=True,
        service="visionops-api",
        product_count=len(state["products"]),
        inspection_count=len(state["inspections"]),
    )


@app.get("/api/twin/state", response_model=TwinSnapshot)
def twin_state() -> TwinSnapshot:
    return twin.snapshot()


@app.post("/api/assistant/ask", response_model=AssistantAnswer)
def ask_dashboard_assistant(payload: AssistantQuestion) -> AssistantAnswer:
    return answer_dashboard_question(payload, store.read(), twin.snapshot())


@app.websocket("/api/twin/ws")
async def twin_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(twin.snapshot().model_dump(mode="json"))
            await asyncio.sleep(0.35)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.get("/api/datasets/catalog")
def dataset_catalog() -> dict:
    if not DATASET_CATALOG.exists():
        return {"datasets": []}
    with DATASET_CATALOG.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/api/sample-inspections")
def sample_inspections(pack_id: str = "visa_capsules") -> dict:
    return _load_sample_pack(pack_id)


@app.get("/api/training/runs", response_model=list[TrainingRun])
def training_runs() -> list[TrainingRun]:
    runs: list[TrainingRun] = []
    for run_dir in sorted(TRAINING_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "train_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        eval_path = run_dir / "evaluation" / "eval_summary.json"
        metrics = None
        if eval_path.exists():
            with eval_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle).get("metrics")
        runs.append(
            TrainingRun(
                id=run_dir.name,
                run_name=summary.get("run_name", run_dir.name),
                source=summary.get("source", "unknown"),
                category=summary.get("category"),
                created_at=summary.get("created_at"),
                model_path=summary.get("model_path"),
                threshold=summary.get("threshold"),
                normal_images=summary.get("normal_images"),
                memory_patches=summary.get("memory_patches"),
                metrics=metrics,
            )
        )

    for run_dir in sorted(ANOMALIB_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics: dict | None = None
        test_results = summary.get("test_results")
        if isinstance(test_results, list) and test_results and isinstance(test_results[0], dict):
            metrics = test_results[0]
        dataset_root = str(summary.get("dataset_root", ""))
        runs.append(
            TrainingRun(
                id=run_dir.name,
                run_name=summary.get("run_name", run_dir.name),
                source=f"anomalib:{summary.get('model', 'unknown')}",
                category="capsules" if "visa_capsules" in dataset_root else None,
                created_at=summary.get("created_at"),
                model_path=summary.get("checkpoint"),
                threshold=None,
                normal_images=None,
                memory_patches=None,
                metrics=metrics,
            )
        )

    return sorted(runs, key=lambda run: (run.created_at or "", run.id), reverse=True)


@app.get("/api/model-versions", response_model=list[ModelVersion])
def model_versions() -> list[ModelVersion]:
    state = store.read()
    versions = [ModelVersion(**item) for item in state.get("model_versions", [])]
    return sorted(versions, key=lambda item: (item.created_at, item.version), reverse=True)


@app.post("/api/model-versions/{version_id}/approve", response_model=ModelVersionActionResponse)
def approve_model_version(version_id: str, approved_by: str = Form("QA demo")) -> ModelVersionActionResponse:
    selected: dict | None = None
    product_snapshot: dict | None = None

    def mutator(state: dict) -> None:
        nonlocal selected, product_snapshot
        version = _find_model_version(state, version_id)
        version["status"] = "approved"
        version["approved_at"] = _now().isoformat()
        version["approved_by"] = approved_by
        selected = version
        product_id = version.get("product_id")
        if product_id:
            product = _find_product(state, product_id)
            product["approved_model_count"] = sum(
                1
                for item in state.get("model_versions", [])
                if item.get("product_id") == product_id and item.get("status") in {"approved", "active"}
            )
            product["updated_at"] = _now().isoformat()
            product_snapshot = product
        _append_audit(
            state,
            _audit_event(
                "model.approved",
                f"Approved model version {version.get('version')} for {version.get('product_name') or 'research run'}.",
                product_id=product_id,
                model_version_id=version_id,
                details={"approved_by": approved_by, "model_kind": version.get("model_kind")},
            ),
        )

    store.update(mutator)
    assert selected is not None
    return ModelVersionActionResponse(
        model_version=ModelVersion(**selected),
        product=Product(**product_snapshot) if product_snapshot else None,
    )


@app.post("/api/model-versions/{version_id}/activate", response_model=ModelVersionActionResponse)
def activate_model_version(version_id: str) -> ModelVersionActionResponse:
    selected: dict | None = None
    product_snapshot: dict | None = None

    def mutator(state: dict) -> None:
        nonlocal selected, product_snapshot
        version = _find_model_version(state, version_id)
        product_id = version.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Only product model versions can be activated for live inspection.")
        product = _find_product(state, product_id)
        for item in state.get("model_versions", []):
            if item.get("product_id") == product_id and item.get("status") == "active":
                item["status"] = "approved"
        version["status"] = "active"
        version["approved_at"] = version.get("approved_at") or _now().isoformat()
        version["approved_by"] = version.get("approved_by") or "QA demo"
        version["activated_at"] = _now().isoformat()
        product["status"] = "ready"
        product["model_path"] = version["model_path"]
        product["threshold"] = version.get("threshold")
        product["model_kind"] = version.get("model_kind")
        product["active_model_version_id"] = version_id
        product["updated_at"] = _now().isoformat()
        selected = version
        product_snapshot = product
        _append_audit(
            state,
            _audit_event(
                "model.activated",
                f"Activated model v{version.get('version')} for {product.get('name')}.",
                product_id=product_id,
                model_version_id=version_id,
            ),
        )

    store.update(mutator)
    assert selected is not None and product_snapshot is not None
    return ModelVersionActionResponse(model_version=ModelVersion(**selected), product=Product(**product_snapshot))


@app.get("/api/products", response_model=list[Product])
def list_products() -> list[Product]:
    return [Product(**product) for product in store.read()["products"]]


@app.post("/api/products", response_model=ProductCreateResponse)
def create_product(
    name: str = Form(...),
    sku: str | None = Form(None),
    product_family: str = Form("capsule"),
    shape: str | None = Form(None),
    diameter_mm: float | None = Form(None),
    length_mm: float | None = Form(None),
    width_mm: float | None = Form(None),
    height_mm: float | None = Form(None),
    cavity_rows: int = Form(4),
    cavity_cols: int = Form(3),
    inspection_sides: int | None = Form(None),
    inspection_channels: str | None = Form("colour,backlight,3d"),
    sorting_mode: str = Form("active_sorting_with_verification"),
    notes: str | None = Form(None),
) -> ProductCreateResponse:
    if cavity_rows < 1 or cavity_cols < 1:
        raise HTTPException(status_code=400, detail="Cavity rows and columns must be positive.")
    if product_family not in {"round_tablet", "oblong_tablet", "capsule", "softgel", "blister_pack"}:
        raise HTTPException(status_code=400, detail="Unsupported product family.")

    timestamp = _now()
    product = Product(
        id=str(uuid.uuid4()),
        name=name.strip(),
        sku=sku.strip() if sku else None,
        product_family=product_family,
        shape=(shape or _default_shape(product_family)).strip(),
        diameter_mm=diameter_mm,
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        cavity_rows=cavity_rows,
        cavity_cols=cavity_cols,
        inspection_sides=_default_inspection_sides(product_family, inspection_sides),
        inspection_channels=_channels_from_form(inspection_channels),
        sorting_mode=sorting_mode.strip() or "active_sorting_with_verification",
        inspection_mode=_default_inspection_mode(product_family),
        notes=notes.strip() if notes else None,
        created_at=timestamp,
        updated_at=timestamp,
    )

    def mutator(state: dict) -> None:
        state["products"].append(product.model_dump(mode="json"))
        _append_audit(
            state,
            _audit_event(
                "product.created",
                f"Created product recipe {product.name}.",
                product_id=product.id,
                details={
                    "sku": product.sku,
                    "family": product.product_family,
                    "shape": product.shape,
                    "layout": f"{product.cavity_rows}x{product.cavity_cols}",
                    "inspection_sides": product.inspection_sides,
                    "inspection_channels": product.inspection_channels,
                    "sorting_mode": product.sorting_mode,
                },
            ),
        )

    store.update(mutator)
    return ProductCreateResponse(product=product)


@app.post("/api/products/{product_id}/teach", response_model=TeachResponse)
def teach_product(product_id: str, files: list[UploadFile] = File(...)) -> TeachResponse:
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="Upload at least 3 good product samples.")

    state = store.read()
    product = _find_product(state, product_id)
    version_number = _next_product_model_version(state, product_id)
    product_dir = MEDIA_DIR / "products" / product_id / "teach"
    image_paths: list[Path] = []
    for upload in files:
        suffix = Path(_safe_name(upload.filename)).suffix or ".png"
        dest = product_dir / f"{uuid.uuid4().hex}{suffix}"
        image_paths.append(_save_upload(upload, dest))

    model_path = MODELS_DIR / product_id / f"normal_reference_v{version_number}.npz"
    try:
        training = train_normal_model(
            image_paths,
            model_path,
            rows=int(product["cavity_rows"]),
            cols=int(product["cavity_cols"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not teach product: {exc}") from exc

    updated_product: dict | None = None

    def mutator(next_state: dict) -> None:
        nonlocal updated_product
        item = _find_product(next_state, product_id)
        version_id = str(uuid.uuid4())
        for version in next_state.get("model_versions", []):
            if version.get("product_id") == product_id and version.get("status") == "active":
                version["status"] = "approved"
        model_version = ModelVersion(
            id=version_id,
            product_id=product_id,
            product_name=item["name"],
            version=version_number,
            model_kind="normal-reference-anomaly",
            status="active",
            model_path=str(model_path),
            threshold=float(training["threshold"]),
            training_samples=int(training["sample_count"]),
            metrics={
                "training_score_min": min(training["training_scores"]),
                "training_score_max": max(training["training_scores"]),
                "training_score_avg": round(float(np.mean(training["training_scores"])), 4),
            },
            created_at=_now(),
            approved_at=_now(),
            approved_by="auto-demo",
            activated_at=_now(),
            notes="Auto-approved for MVP demo; production should require QA review.",
        ).model_dump(mode="json")
        next_state.setdefault("model_versions", []).insert(0, model_version)
        item["status"] = "ready"
        item["good_sample_count"] = int(training["sample_count"])
        item["threshold"] = float(training["threshold"])
        item["model_path"] = str(model_path)
        item["active_model_version_id"] = version_id
        item["model_kind"] = "normal-reference-anomaly"
        item["approved_model_count"] = sum(
            1
            for version in next_state.get("model_versions", [])
            if version.get("product_id") == product_id and version.get("status") in {"approved", "active"}
        )
        item["updated_at"] = _now().isoformat()
        updated_product = item
        _append_audit(
            next_state,
            _audit_event(
                "model.trained",
                f"Trained and activated model v{version_number} for {item['name']}.",
                product_id=product_id,
                model_version_id=version_id,
                details={
                    "samples": int(training["sample_count"]),
                    "threshold": round(float(training["threshold"]), 4),
                    "model_kind": "normal-reference-anomaly",
                },
            ),
        )

    store.update(mutator)
    assert updated_product is not None
    return TeachResponse(
        product=Product(**updated_product),
        training_scores=training["training_scores"],
        message="Product recipe is ready for inspection.",
    )


def _inspect_uploaded_file(product_id: str, product: dict, file: UploadFile, batch_id: str | None = None) -> Inspection:
    started = time.perf_counter()
    inspection_id = str(uuid.uuid4())
    source_name = _safe_name(file.filename)
    active_batch_id = batch_id or "B-2026-017"
    twin.begin_inspection(product, active_batch_id, source_name)
    suffix = Path(source_name).suffix or ".png"
    inspection_dir = MEDIA_DIR / "inspections" / inspection_id
    source_path = _save_upload(file, inspection_dir / f"source{suffix}")
    heatmap_path = inspection_dir / "heatmap.png"

    try:
        model_path = _resolve_model_path(str(product["model_path"]))
        model_kind = str(product.get("model_kind") or "")
        if model_kind.startswith("patchcore-lite"):
            result = inspect_with_patchcore_lite_model(model_path, source_path)
        else:
            expects_blister_grid = product.get("product_family", "blister_pack") == "blister_pack"
            if expects_blister_grid and ("capsule" in source_name.lower() or _looks_like_loose_capsule_image(source_path)):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "This looks like a loose capsule image. Select the Capsule Blister Demo recipe; "
                        "the selected blister-grid recipe expects a 4x3 packaged blister image."
                    ),
                )
            result = inspect_with_model(model_path, source_path)
        result.heatmap.save(heatmap_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not inspect image: {exc}") from exc

    return Inspection(
        id=inspection_id,
        product_id=product_id,
        product_name=product["name"],
        source_name=source_name,
        batch_id=active_batch_id,
        model_version_id=product.get("active_model_version_id"),
        decision=result.decision,  # type: ignore[arg-type]
        score=result.score,
        threshold=result.threshold,
        defect_regions=result.defect_regions,
        image_url=_to_media_url(source_path),
        heatmap_url=_to_media_url(heatmap_path),
        created_at=_now(),
        duration_ms=int((time.perf_counter() - started) * 1000),
        timings_ms=result.timings_ms,
        model_cache_hit=result.model_cache_hit,
    )


@app.post("/api/inspect/upload", response_model=InspectionResponse)
def inspect_upload(
    product_id: str = Form(...),
    file: UploadFile = File(...),
    batch_id: str | None = Form(None),
) -> InspectionResponse:
    state = store.read()
    product = _find_product(state, product_id)
    if product.get("status") != "ready" or not product.get("model_path"):
        raise HTTPException(status_code=400, detail="Teach this product with good samples before inspection.")

    inspection = _inspect_uploaded_file(product_id, product, file, batch_id)

    def mutator(next_state: dict) -> None:
        next_state["inspections"].insert(0, inspection.model_dump(mode="json"))
        next_state["inspections"] = next_state["inspections"][:250]
        _append_audit(
            next_state,
            _audit_event(
                "inspection.completed",
                f"{inspection.decision.upper()} inspection for {inspection.product_name}: score {inspection.score}.",
                product_id=product_id,
                inspection_id=inspection.id,
                model_version_id=inspection.model_version_id,
                details={
                    "batch_id": inspection.batch_id,
                    "duration_ms": inspection.duration_ms,
                    "cache_hit": inspection.model_cache_hit,
                    "defect_types": [region.defect_type for region in inspection.defect_regions],
                },
            ),
        )

    store.update(mutator)
    twin.complete_inspection(inspection, batch_position=1, batch_size=1)
    return InspectionResponse(inspection=inspection)


@app.post("/api/inspect/batch", response_model=list[Inspection])
def inspect_batch(
    product_id: str = Form(...),
    files: list[UploadFile] = File(...),
    batch_id: str | None = Form(None),
) -> list[Inspection]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > 25:
        raise HTTPException(status_code=400, detail="Batch inspection is limited to 25 images in the MVP.")
    state = store.read()
    product = _find_product(state, product_id)
    if product.get("status") != "ready" or not product.get("model_path"):
        raise HTTPException(status_code=400, detail="Teach this product with good samples before inspection.")
    batch_label = batch_id or "B-2026-017"
    inspections = []
    batch_size = len(files)
    for batch_index, file in enumerate(files, start=1):
        inspection = _inspect_uploaded_file(product_id, product, file, batch_label)
        twin.complete_inspection(inspection, batch_position=batch_index, batch_size=batch_size)
        inspections.append(inspection)

    def mutator(next_state: dict) -> None:
        next_state["inspections"] = [item.model_dump(mode="json") for item in inspections] + next_state["inspections"]
        next_state["inspections"] = next_state["inspections"][:250]
        decisions = {decision: sum(1 for item in inspections if item.decision == decision) for decision in ("pass", "review", "reject")}
        _append_audit(
            next_state,
            _audit_event(
                "batch.completed",
                f"Completed batch {batch_label}: {len(inspections)} images.",
                product_id=product_id,
                details={"batch_id": batch_label, "decisions": decisions},
            ),
        )

    store.update(mutator)
    return inspections


@app.post("/api/inspect/samples", response_model=list[Inspection])
def inspect_sample_batch(payload: dict) -> list[Inspection]:
    product_id = str(payload.get("product_id") or "")
    sample_ids = [str(item) for item in payload.get("sample_ids", [])]
    batch_id = str(payload.get("batch_id") or "B-2026-DEMO")
    if not product_id:
        raise HTTPException(status_code=400, detail="Choose a ready product before running sample inspections.")
    if not sample_ids:
        raise HTTPException(status_code=400, detail="Select at least one normal or anomaly sample.")
    if len(sample_ids) > 25:
        raise HTTPException(status_code=400, detail="Sample inspection is limited to 25 images.")

    state = store.read()
    product = _find_product(state, product_id)
    if product.get("status") != "ready" or not product.get("model_path"):
        raise HTTPException(status_code=400, detail="Teach this product with good samples before inspection.")

    inspections = []
    batch_size = len(sample_ids)
    for batch_index, sample_id in enumerate(sample_ids, start=1):
        record, image_path = _resolve_sample_record(sample_id)
        with image_path.open("rb") as sample_file:
            upload = UploadFile(file=sample_file, filename=Path(record["image_url"]).name)
            inspection = _inspect_uploaded_file(product_id, product, upload, batch_id)
        twin.complete_inspection(inspection, batch_position=batch_index, batch_size=batch_size)
        inspections.append(inspection)

    def mutator(next_state: dict) -> None:
        next_state["inspections"] = [item.model_dump(mode="json") for item in inspections] + next_state["inspections"]
        next_state["inspections"] = next_state["inspections"][:250]
        decisions = {decision: sum(1 for item in inspections if item.decision == decision) for decision in ("pass", "review", "reject")}
        _append_audit(
            next_state,
            _audit_event(
                "demo_batch.completed",
                f"Completed sample demo batch {batch_id}: {len(inspections)} images.",
                product_id=product_id,
                details={"batch_id": batch_id, "sample_ids": sample_ids, "decisions": decisions},
            ),
        )

    store.update(mutator)
    return inspections


@app.get("/api/inspections", response_model=list[Inspection])
def list_inspections(limit: int = 50) -> list[Inspection]:
    limit = min(max(limit, 1), 250)
    return [Inspection(**item) for item in store.read()["inspections"][:limit]]


@app.get("/api/audit/events", response_model=list[AuditEvent])
def audit_events(limit: int = 80) -> list[AuditEvent]:
    limit = min(max(limit, 1), 500)
    return [AuditEvent(**item) for item in store.read().get("audit_events", [])[:limit]]


@app.get("/api/reports/batch", response_model=BatchReport)
def batch_report(product_id: str | None = None, limit: int = 250) -> BatchReport:
    state = store.read()
    limit = min(max(limit, 1), 250)
    records = state.get("inspections", [])[:limit]
    if product_id:
        records = [item for item in records if item.get("product_id") == product_id]
    inspections = [Inspection(**item) for item in records]
    product_name = None
    if product_id:
        try:
            product_name = _find_product(state, product_id).get("name")
        except HTTPException:
            product_name = None

    total = len(inspections)
    pass_count = sum(1 for item in inspections if item.decision == "pass")
    review_count = sum(1 for item in inspections if item.decision == "review")
    reject_count = sum(1 for item in inspections if item.decision == "reject")
    durations = [float(item.duration_ms) for item in inspections]
    scores = [float(item.score) for item in inspections]
    defect_type_counts: dict[str, int] = {}
    for inspection in inspections:
        for region in inspection.defect_regions:
            defect_type_counts[region.defect_type] = defect_type_counts.get(region.defect_type, 0) + 1

    return BatchReport(
        product_id=product_id,
        product_name=product_name,
        total_inspections=total,
        pass_count=pass_count,
        review_count=review_count,
        reject_count=reject_count,
        reject_rate=round(reject_count / total, 4) if total else 0.0,
        avg_score=round(float(np.mean(scores)), 4) if scores else 0.0,
        avg_duration_ms=round(float(np.mean(durations)), 3) if durations else 0.0,
        p95_duration_ms=_percentile(durations, 95),
        defect_type_counts=defect_type_counts,
        generated_at=_now(),
    )


@app.get("/api/performance", response_model=PerformanceStats)
def performance_stats(limit: int = 250) -> PerformanceStats:
    limit = min(max(limit, 1), 250)
    inspections = [Inspection(**item) for item in store.read().get("inspections", [])[:limit]]
    durations = [float(item.duration_ms) for item in inspections]
    inference = [float(item.timings_ms.get("inference", 0.0)) for item in inspections]
    decision_counts = {decision: sum(1 for item in inspections if item.decision == decision) for decision in ("pass", "review", "reject")}
    return PerformanceStats(
        total_inspections=len(inspections),
        avg_duration_ms=round(float(np.mean(durations)), 3) if durations else 0.0,
        p50_duration_ms=_percentile(durations, 50),
        p95_duration_ms=_percentile(durations, 95),
        avg_inference_ms=round(float(np.mean(inference)), 3) if inference else 0.0,
        cache_hit_rate=round(sum(1 for item in inspections if item.model_cache_hit) / len(inspections), 4) if inspections else 0.0,
        decision_counts=decision_counts,
    )
