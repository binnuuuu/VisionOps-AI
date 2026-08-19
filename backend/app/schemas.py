from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Decision = Literal["pass", "review", "reject"]
ProductStatus = Literal["draft", "ready"]
ModelVersionStatus = Literal["candidate", "approved", "active", "archived"]
ProductFamily = Literal["round_tablet", "oblong_tablet", "capsule", "softgel", "blister_pack"]
TwinAssetType = Literal[
    "feeder",
    "camera",
    "inspection_unit",
    "ai_decision_unit",
    "conveyor",
    "actuator",
    "sorting_bin",
    "operator_panel",
]


class Product(BaseModel):
    id: str
    name: str
    sku: str | None = None
    product_family: ProductFamily = "blister_pack"
    shape: str = "blister_pack"
    diameter_mm: float | None = Field(default=None, ge=0, le=100)
    length_mm: float | None = Field(default=None, ge=0, le=100)
    width_mm: float | None = Field(default=None, ge=0, le=100)
    height_mm: float | None = Field(default=None, ge=0, le=100)
    cavity_rows: int = Field(ge=1, le=20)
    cavity_cols: int = Field(ge=1, le=20)
    inspection_sides: int = Field(default=6, ge=1, le=6)
    inspection_channels: list[str] = Field(default_factory=lambda: ["colour", "backlight", "3d"])
    sorting_mode: str = "active_sorting_with_verification"
    inspection_mode: str = "loose_product"
    notes: str | None = None
    status: ProductStatus = "draft"
    good_sample_count: int = 0
    threshold: float | None = None
    model_path: str | None = None
    active_model_version_id: str | None = None
    model_kind: str | None = None
    approved_model_count: int = 0
    created_at: datetime
    updated_at: datetime


class ProductCreateResponse(BaseModel):
    product: Product


class TeachResponse(BaseModel):
    product: Product
    training_scores: list[float]
    message: str


class DefectRegion(BaseModel):
    cavity: int
    row: int
    col: int
    score: float
    bbox: list[int]
    defect_type: str = "unclassified_deviation"
    severity: str = "moderate"
    confidence: float = 0.0
    area_pct: float = 0.0


class Inspection(BaseModel):
    id: str
    product_id: str
    product_name: str
    source_name: str
    batch_id: str | None = None
    model_version_id: str | None = None
    decision: Decision
    score: float
    threshold: float
    defect_regions: list[DefectRegion]
    image_url: str
    heatmap_url: str
    created_at: datetime
    duration_ms: int
    timings_ms: dict[str, float] = Field(default_factory=dict)
    model_cache_hit: bool = False


class InspectionResponse(BaseModel):
    inspection: Inspection


class HealthResponse(BaseModel):
    ok: bool
    service: str
    product_count: int
    inspection_count: int


class TrainingRun(BaseModel):
    id: str
    run_name: str
    source: str
    category: str | None = None
    created_at: str | None = None
    model_path: str | None = None
    threshold: float | None = None
    normal_images: int | None = None
    memory_patches: int | None = None
    metrics: dict | None = None


class ModelVersion(BaseModel):
    id: str
    product_id: str | None = None
    product_name: str | None = None
    version: int
    model_kind: str
    status: ModelVersionStatus
    model_path: str
    threshold: float | None = None
    training_samples: int | None = None
    metrics: dict[str, Any] | None = None
    created_at: datetime
    approved_at: datetime | None = None
    approved_by: str | None = None
    activated_at: datetime | None = None
    notes: str | None = None


class ModelVersionActionResponse(BaseModel):
    model_version: ModelVersion
    product: Product | None = None


class AuditEvent(BaseModel):
    id: str
    event_type: str
    message: str
    product_id: str | None = None
    inspection_id: str | None = None
    model_version_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class BatchReport(BaseModel):
    product_id: str | None = None
    product_name: str | None = None
    total_inspections: int
    pass_count: int
    review_count: int
    reject_count: int
    reject_rate: float
    avg_score: float
    avg_duration_ms: float
    p95_duration_ms: float
    defect_type_counts: dict[str, int]
    generated_at: datetime


class PerformanceStats(BaseModel):
    total_inspections: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    avg_inference_ms: float
    cache_hit_rate: float
    decision_counts: dict[str, int]
    target_latency_ms: int = 50


class AssistantQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    product_id: str | None = None
    inspection_id: str | None = None
    batch_id: str | None = None


class AssistantAnswer(BaseModel):
    answer: str
    intent: str
    context: dict[str, Any] = Field(default_factory=dict)
    suggested_questions: list[str] = Field(default_factory=list)
    generated_at: datetime


class TwinPlant(BaseModel):
    plant_id: str
    name: str
    status: str
    location: str


class TwinInspectionLine(BaseModel):
    line_id: str
    plant_id: str
    name: str
    status: str
    operating_mode: str


class TwinAsset(BaseModel):
    asset_id: str
    line_id: str
    asset_type: TwinAssetType
    name: str
    status: str
    current_recipe: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TwinRecipe(BaseModel):
    recipe_id: str
    name: str
    product_family: str
    inspection_sides: int
    inspection_channels: list[str]


class TwinBatch(BaseModel):
    batch_id: str
    status: str
    inspected_count: int
    good_count: int
    review_count: int
    reject_count: int


class TwinModel(BaseModel):
    model_id: str
    name: str
    status: str
    model_kind: str
    threshold: float | None = None


class TwinInspectionEvent(BaseModel):
    event_id: str
    event_type: str
    message: str
    created_at: datetime
    decision: Decision | None = None
    asset_id: str | None = None
    inspection_id: str | None = None
    defect_type: str | None = None
    score: float | None = None
    evidence_image_url: str | None = None
    source_image_url: str | None = None
    source_name: str | None = None
    batch_id: str | None = None
    batch_position: int | None = None
    batch_size: int | None = None


class TwinCounters(BaseModel):
    good: int
    review: int
    reject: int


class TwinCycle(BaseModel):
    inspection_id: str
    source_name: str
    product_name: str
    batch_id: str
    batch_position: int
    batch_size: int
    queue_depth: int
    phase: str
    status_message: str
    progress_pct: float
    decision: Decision | None = None
    score: float
    threshold: float
    defect_type: str | None = None
    source_image_url: str
    evidence_image_url: str
    started_at: datetime


class TwinSnapshot(BaseModel):
    sequence: int
    updated_at: datetime
    connection_state: str
    plant: TwinPlant
    inspection_line: TwinInspectionLine
    assets: list[TwinAsset]
    recipe: TwinRecipe
    batch: TwinBatch
    model: TwinModel
    counters: TwinCounters
    route_state: str
    pulse_active: bool
    active_cycle: TwinCycle | None = None
    last_event: TwinInspectionEvent | None = None
    timeline: list[TwinInspectionEvent]
