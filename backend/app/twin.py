from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    TwinAsset,
    TwinBatch,
    TwinCounters,
    TwinCycle,
    TwinInspectionEvent,
    TwinInspectionLine,
    TwinModel,
    TwinPlant,
    TwinRecipe,
    TwinSnapshot,
)


CYCLE_SECONDS = 5.8
DECISION_PROGRESS = 0.62


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DigitalTwin:
    """Backend-owned operating twin driven by ordered inspection cycles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 1
        self._last_source_name = ""
        self._counters = {"good": 1842, "review": 38, "reject": 126}
        self._pending_cycles: deque[dict[str, Any]] = deque()
        self._active_cycle: dict[str, Any] | None = None
        self._recipe = TwinRecipe(
            recipe_id="CAPSULE_BLUE_WHITE_01",
            name="Blue / White Capsule",
            product_family="capsule",
            inspection_sides=6,
            inspection_channels=["colour", "backlight", "3d"],
        )
        self._batch = TwinBatch(
            batch_id="B-2026-017",
            status="running",
            inspected_count=sum(self._counters.values()),
            good_count=self._counters["good"],
            review_count=self._counters["review"],
            reject_count=self._counters["reject"],
        )
        self._model = TwinModel(
            model_id="MODEL_PATCHCORE_CAPSULE_01",
            name="Capsule Surface Anomaly Model",
            status="active",
            model_kind="patchcore-lite",
            threshold=0.42,
        )
        self._last_event: TwinInspectionEvent | None = None
        self._timeline = [
            TwinInspectionEvent(
                event_id=str(uuid.uuid4()),
                event_type="batch.started",
                message="Batch B-2026-017 released to inspection line.",
                created_at=_now(),
                asset_id="LINE_01",
            ),
            TwinInspectionEvent(
                event_id=str(uuid.uuid4()),
                event_type="model.activated",
                message="Capsule anomaly model active and recipe verified.",
                created_at=_now(),
                asset_id="AI_01",
            ),
            TwinInspectionEvent(
                event_id=str(uuid.uuid4()),
                event_type="line.ready",
                message="Inspection line running in automatic mode.",
                created_at=_now(),
                asset_id="LINE_01",
            ),
        ]

    @staticmethod
    def _event_from_record(record: dict[str, Any], *, restored: bool = False) -> TwinInspectionEvent:
        regions = list(record.get("defect_regions") or [])
        first_region = regions[0] if regions else None
        defect_type = first_region.get("defect_type") if isinstance(first_region, dict) else None
        decision = str(record.get("decision") or "review")
        message = {
            "reject": f"Defect detected: {defect_type or 'visual anomaly'} routed to reject bin.",
            "review": f"Uncertain product routed to operator review: {defect_type or 'threshold warning'}.",
            "pass": "Product accepted and routed to the good outlet.",
        }.get(decision, "Inspection event completed.")
        return TwinInspectionEvent(
            event_id=f"restored-{record.get('id')}" if restored else str(uuid.uuid4()),
            event_type={
                "reject": "defect.detected",
                "review": "inspection.review_required",
                "pass": "inspection.passed",
            }.get(decision, "inspection.completed"),
            message=message,
            created_at=record.get("created_at") or _now(),
            decision=decision,
            asset_id="INSPECTION_01" if decision == "reject" else "AI_01",
            inspection_id=str(record.get("id") or ""),
            defect_type=defect_type,
            score=record.get("score"),
            evidence_image_url=record.get("heatmap_url"),
            source_image_url=record.get("image_url"),
            source_name=record.get("source_name"),
            batch_id=record.get("batch_id"),
            batch_position=record.get("batch_position"),
            batch_size=record.get("batch_size"),
        )

    def _set_product_context_locked(self, product: dict[str, Any], batch_id: str) -> None:
        self._recipe = TwinRecipe(
            recipe_id=str(product.get("sku") or product.get("id") or "RECIPE_ACTIVE"),
            name=str(product.get("name") or "Active product recipe"),
            product_family=str(product.get("product_family") or "capsule"),
            inspection_sides=int(product.get("inspection_sides") or 6),
            inspection_channels=list(product.get("inspection_channels") or ["colour", "backlight", "3d"]),
        )
        self._batch = self._batch.model_copy(update={"batch_id": batch_id, "status": "running"})
        self._model = TwinModel(
            model_id=str(product.get("active_model_version_id") or "MODEL_ACTIVE"),
            name=f"{product.get('name') or 'Product'} anomaly model",
            status="active",
            model_kind=str(product.get("model_kind") or "visual-anomaly"),
            threshold=product.get("threshold"),
        )

    def hydrate(self, state: dict[str, Any]) -> None:
        """Restore the latest recipe and evidence without replaying actuators."""
        inspections = list(state.get("inspections") or [])
        if not inspections:
            return
        latest = inspections[0]
        products = {str(product.get("id")): product for product in state.get("products") or []}
        product = products.get(str(latest.get("product_id")))
        event = self._event_from_record(latest, restored=True)
        with self._lock:
            if product:
                self._set_product_context_locked(product, str(latest.get("batch_id") or self._batch.batch_id))
            self._last_source_name = str(latest.get("source_name") or "")
            self._last_event = event
            self._timeline.insert(0, event)

    def begin_inspection(self, product: dict[str, Any], batch_id: str, source_name: str) -> None:
        with self._lock:
            self._sequence += 1
            self._last_source_name = source_name
            self._set_product_context_locked(product, batch_id)

    def _activate_next_locked(self, now_mono: float) -> None:
        if self._active_cycle is not None or not self._pending_cycles:
            return
        self._active_cycle = self._pending_cycles.popleft()
        self._active_cycle["started_mono"] = now_mono
        self._active_cycle["started_at"] = _now()
        self._active_cycle["announced"] = False
        self._sequence += 1

    def _announce_cycle_locked(self, cycle: dict[str, Any]) -> None:
        if cycle["announced"]:
            return
        cycle["announced"] = True
        decision = str(cycle["decision"])
        counter_key = decision if decision in self._counters else "review"
        self._counters[counter_key] += 1
        self._batch = TwinBatch(
            batch_id=str(cycle["batch_id"] or self._batch.batch_id),
            status="running",
            inspected_count=self._batch.inspected_count + 1,
            good_count=self._counters["good"],
            review_count=self._counters["review"],
            reject_count=self._counters["reject"],
        )
        event = cycle["event"]
        self._last_event = event
        self._timeline.insert(0, event)
        self._timeline = self._timeline[:24]
        self._sequence += 1

    def _advance_cycle_locked(self, now_mono: float) -> tuple[dict[str, Any] | None, float]:
        self._activate_next_locked(now_mono)
        if self._active_cycle is None:
            return None, 0.0

        elapsed = now_mono - float(self._active_cycle["started_mono"])
        progress = min(max(elapsed / CYCLE_SECONDS, 0.0), 1.0)
        if progress >= DECISION_PROGRESS:
            self._announce_cycle_locked(self._active_cycle)
        if progress >= 1.0:
            self._active_cycle = None
            self._activate_next_locked(now_mono)
            if self._active_cycle is None:
                return None, 0.0
            return self._active_cycle, 0.0
        return self._active_cycle, progress

    def complete_inspection(self, inspection: Any, *, batch_position: int = 1, batch_size: int = 1) -> None:
        record = inspection.model_dump(mode="json")
        record["batch_position"] = batch_position
        record["batch_size"] = batch_size
        event = self._event_from_record(record)
        regions = list(inspection.defect_regions)
        defect_type = regions[0].defect_type if regions else None
        cycle = {
            "inspection_id": str(inspection.id),
            "source_name": str(inspection.source_name),
            "product_name": str(inspection.product_name),
            "batch_id": str(inspection.batch_id or self._batch.batch_id),
            "batch_position": batch_position,
            "batch_size": batch_size,
            "decision": str(inspection.decision),
            "score": float(inspection.score),
            "threshold": float(inspection.threshold),
            "defect_type": defect_type,
            "source_image_url": str(inspection.image_url),
            "evidence_image_url": str(inspection.heatmap_url),
            "event": event,
        }
        with self._lock:
            self._pending_cycles.append(cycle)
            self._activate_next_locked(time.monotonic())
            self._sequence += 1

    @staticmethod
    def _phase(progress: float) -> tuple[str, str]:
        if progress < 0.16:
            return "feeding", "Entering conveyor"
        if progress < 0.34:
            return "capture", "Acquiring camera frame"
        if progress < DECISION_PROGRESS:
            return "inspection", "AI inference and localization"
        if progress < 0.80:
            return "decision", "Decision verified"
        return "sorting", "Routing to verified outlet"

    def snapshot(self) -> TwinSnapshot:
        with self._lock:
            now_mono = time.monotonic()
            now = _now()
            cycle, progress = self._advance_cycle_locked(now_mono)
            phase, status_message = self._phase(progress) if cycle else ("idle", "Waiting for product")
            decision_visible = cycle is not None and progress >= DECISION_PROGRESS
            route_state = str(cycle["decision"]) if decision_visible else ("processing" if cycle else "normal")
            camera_status = "capturing" if phase == "capture" else "online"
            if decision_visible and cycle["decision"] != "pass" and phase in {"decision", "sorting"}:
                camera_status = "alert"
            inspection_status = "processing" if phase in {"capture", "inspection"} else "ready"
            if decision_visible and cycle["decision"] != "pass":
                inspection_status = "warning"
            ai_status = "inferencing" if phase in {"inspection", "decision"} else "active"
            gate_status = "activated" if phase == "sorting" and cycle and cycle["decision"] == "reject" else "armed"
            source_name = str(cycle["source_name"]) if cycle else self._last_source_name
            camera_metrics = {
                "fps": 24,
                "last_frame_time": now.isoformat(),
                "source": source_name or "live stream",
            }
            assets = [
                TwinAsset(
                    asset_id="FEEDER_01",
                    line_id="LINE_01",
                    asset_type="feeder",
                    name="Tablet Feeder",
                    status="running",
                    current_recipe=self._recipe.recipe_id,
                    metrics={"rate_per_min": 184, "hopper_level_pct": 72},
                ),
                TwinAsset(
                    asset_id="CAMERA_01",
                    line_id="LINE_01",
                    asset_type="camera",
                    name="Top Inspection Camera",
                    status=camera_status,
                    current_recipe=self._recipe.recipe_id,
                    metrics=camera_metrics,
                ),
                TwinAsset(
                    asset_id="INSPECTION_01",
                    line_id="LINE_01",
                    asset_type="inspection_unit",
                    name="Six-Side Inspection Station",
                    status=inspection_status,
                    current_recipe=self._recipe.recipe_id,
                    metrics={"inspection_sides": self._recipe.inspection_sides, "channels": self._recipe.inspection_channels},
                ),
                TwinAsset(
                    asset_id="AI_01",
                    line_id="LINE_01",
                    asset_type="ai_decision_unit",
                    name="AI Decision Unit",
                    status=ai_status,
                    current_recipe=self._recipe.recipe_id,
                    metrics={"model_id": self._model.model_id, "threshold": self._model.threshold},
                ),
                TwinAsset(
                    asset_id="CONVEYOR_01",
                    line_id="LINE_01",
                    asset_type="conveyor",
                    name="Product Conveyor",
                    status="running",
                    current_recipe=self._recipe.recipe_id,
                    metrics={"speed_m_min": 12.4},
                ),
                TwinAsset(
                    asset_id="ACTUATOR_REJECT_01",
                    line_id="LINE_01",
                    asset_type="actuator",
                    name="Reject Gate",
                    status=gate_status,
                    current_recipe=self._recipe.recipe_id,
                    metrics={"cycle_time_ms": 86, "verified": True},
                ),
                TwinAsset(
                    asset_id="BIN_GOOD_01",
                    line_id="LINE_01",
                    asset_type="sorting_bin",
                    name="Good Bin",
                    status="available",
                    metrics={"count": self._counters["good"], "fill_pct": round(self._counters["good"] / 50, 1)},
                ),
                TwinAsset(
                    asset_id="BIN_REVIEW_01",
                    line_id="LINE_01",
                    asset_type="sorting_bin",
                    name="Review Bin",
                    status="available",
                    metrics={"count": self._counters["review"], "fill_pct": round(self._counters["review"] / 2, 1)},
                ),
                TwinAsset(
                    asset_id="BIN_REJECT_01",
                    line_id="LINE_01",
                    asset_type="sorting_bin",
                    name="Reject Bin",
                    status="available",
                    metrics={"count": self._counters["reject"], "fill_pct": round(self._counters["reject"] / 2, 1)},
                ),
                TwinAsset(
                    asset_id="HMI_01",
                    line_id="LINE_01",
                    asset_type="operator_panel",
                    name="Operator Panel",
                    status="online",
                    current_recipe=self._recipe.recipe_id,
                    metrics={
                        "session": "operator",
                        "alarms": 1 if inspection_status == "warning" else 0,
                        "queue_depth": len(self._pending_cycles),
                    },
                ),
            ]

            active_cycle = None
            if cycle:
                active_cycle = TwinCycle(
                    inspection_id=cycle["inspection_id"],
                    source_name=cycle["source_name"],
                    product_name=cycle["product_name"],
                    batch_id=cycle["batch_id"],
                    batch_position=cycle["batch_position"],
                    batch_size=cycle["batch_size"],
                    queue_depth=len(self._pending_cycles),
                    phase=phase,
                    status_message=status_message,
                    progress_pct=round(progress * 100.0, 1),
                    decision=cycle["decision"] if decision_visible else None,
                    score=cycle["score"],
                    threshold=cycle["threshold"],
                    defect_type=cycle["defect_type"] if decision_visible else None,
                    source_image_url=cycle["source_image_url"],
                    evidence_image_url=cycle["evidence_image_url"],
                    started_at=cycle["started_at"],
                )

            return TwinSnapshot(
                sequence=self._sequence,
                updated_at=now,
                connection_state="live",
                plant=TwinPlant(
                    plant_id="PLANT_01",
                    name="Pharma Packaging Plant",
                    status="operational",
                    location="Inspection Hall A",
                ),
                inspection_line=TwinInspectionLine(
                    line_id="LINE_01",
                    plant_id="PLANT_01",
                    name="Tablet and Capsule Inspection Line 01",
                    status="running",
                    operating_mode="automatic",
                ),
                assets=assets,
                recipe=self._recipe,
                batch=self._batch,
                model=self._model,
                counters=TwinCounters(**self._counters),
                route_state=route_state,
                pulse_active=cycle is not None,
                active_cycle=active_cycle,
                last_event=self._last_event,
                timeline=list(self._timeline),
            )
