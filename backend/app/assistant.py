from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from .schemas import AssistantAnswer, AssistantQuestion, TwinSnapshot


DEFAULT_SUGGESTIONS = [
    "What is happening right now?",
    "Why was the latest item rejected?",
    "How many products passed and failed in this batch?",
    "Are the camera and AI model online?",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _label(value: str | None) -> str:
    labels = {
        "visual_anomaly": "unclassified visual deviation",
        "unclassified_deviation": "unclassified visual deviation",
        "broken_capsule_or_leak": "broken capsule or leakage",
        "hair_or_fiber_contamination": "hair or fibre contamination",
        "bubble_or_fill_void": "bubble or fill void",
        "surface_particle_or_stain": "surface particle or stain",
        "capsule_surface_deviation": "capsule surface deviation",
    }
    return labels.get(value or "", (value or "unknown defect").replace("_", " "))


def _count_decisions(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        decision: sum(1 for item in records if item.get("decision") == decision)
        for decision in ("pass", "review", "reject")
    }


def _defect_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in records:
        for region in item.get("defect_regions") or []:
            counts[_label(region.get("defect_type"))] += 1
    return counts


def _format_defects(counts: Counter[str], limit: int = 3) -> str:
    top = counts.most_common(limit)
    if not top:
        return "no classified defect regions"
    return ", ".join(f"{name} ({count})" for name, count in top)


def _asset(snapshot: TwinSnapshot, asset_id: str) -> dict[str, Any] | None:
    return next((item.model_dump(mode="json") for item in snapshot.assets if item.asset_id == asset_id), None)


def _answer_rejection(
    record: dict[str, Any] | None,
    scope_records: list[dict[str, Any]],
    batch_id: str | None,
    aggregate_batch: bool,
) -> str:
    rejected = [item for item in scope_records if item.get("decision") == "reject"]
    if aggregate_batch and len(scope_records) > 1 and rejected:
        counts = _defect_counts(rejected)
        scope = f"Batch {batch_id}" if batch_id else "The current selection"
        return (
            f"{scope} has {len(rejected)} rejected item{'s' if len(rejected) != 1 else ''} out of {len(scope_records)}. "
            f"The recorded rejection evidence is {_format_defects(counts)}. Each rejected image produced an anomaly score above "
            "the recipe's reject boundary, so the sorter routed it to the reject bin."
        )

    target = record if record and record.get("decision") == "reject" else (rejected[0] if rejected else record)
    if not target:
        return "There is no rejected inspection in the current dashboard context. Run or select an inspection and ask again."
    if target.get("decision") != "reject":
        return (
            f"The selected item was not rejected; it was classified as {target.get('decision', 'unknown')}. "
            f"Its anomaly score was {float(target.get('score', 0)):.3f} against a recipe threshold of "
            f"{float(target.get('threshold', 0)):.3f}."
        )
    defects = _defect_counts([target])
    return (
        f"{target.get('source_name', 'The latest item')} was rejected because the model found {_format_defects(defects)}. "
        f"Its anomaly score was {float(target.get('score', 0)):.3f}, above the reject boundary derived from the recipe "
        f"threshold of {float(target.get('threshold', 0)):.3f}. The reject gate was therefore commanded to route it away from good product."
    )


def answer_dashboard_question(
    payload: AssistantQuestion,
    state: dict[str, Any],
    snapshot: TwinSnapshot,
) -> AssistantAnswer:
    question = " ".join(payload.question.strip().lower().split())
    inspections = list(state.get("inspections") or [])
    products = list(state.get("products") or [])

    selected_inspection = next(
        (item for item in inspections if payload.inspection_id and item.get("id") == payload.inspection_id),
        None,
    )
    selected_product = next(
        (item for item in products if payload.product_id and item.get("id") == payload.product_id),
        None,
    )
    if selected_inspection is None and payload.product_id:
        selected_inspection = next((item for item in inspections if item.get("product_id") == payload.product_id), None)
    if selected_product is None and selected_inspection:
        selected_product = next((item for item in products if item.get("id") == selected_inspection.get("product_id")), None)
    if selected_product is None:
        selected_product = next((item for item in products if item.get("status") == "ready"), products[0] if products else None)

    batch_id = payload.batch_id or (selected_inspection or {}).get("batch_id") or snapshot.batch.batch_id
    batch_records = [item for item in inspections if batch_id and item.get("batch_id") == batch_id]
    product_records = [
        item for item in inspections if selected_product and item.get("product_id") == selected_product.get("id")
    ]
    scope_records = batch_records or product_records or inspections
    if selected_inspection is None:
        selected_inspection = scope_records[0] if scope_records else None

    context = {
        "batch_id": batch_id,
        "product_id": (selected_product or {}).get("id"),
        "inspection_id": (selected_inspection or {}).get("id"),
        "twin_sequence": snapshot.sequence,
        "data_updated_at": snapshot.updated_at.isoformat(),
    }
    intent = "dashboard_summary"

    if any(term in question for term in ("hello", "hi ", "hey", "good morning", "good afternoon")):
        intent = "greeting"
        answer = (
            "Hello. I am the VisionOps assistant. I can explain the live inspection state, batch counts, rejection reasons, "
            "defect evidence, machine status, recipes, models, and performance shown on this dashboard."
        )
    elif any(term in question for term in ("what can you", "help", "sample question")):
        intent = "capabilities"
        answer = (
            "I can answer questions grounded in this dashboard: what the line is doing, how many items passed or failed, "
            "why an item was rejected, which defects are common, whether equipment is online, and which recipe or model is active."
        )
    elif any(term in question for term in ("why", "reason", "cause")) and any(
        term in question for term in ("reject", "fail", "failed")
    ):
        intent = "rejection_reason"
        answer = _answer_rejection(selected_inspection, scope_records, batch_id, "batch" in question)
    elif any(term in question for term in ("camera", "conveyor", "feeder", "gate", "bin", "machine", "equipment")):
        intent = "asset_status"
        camera = _asset(snapshot, "CAMERA_01") or {}
        feeder = _asset(snapshot, "FEEDER_01") or {}
        conveyor = _asset(snapshot, "CONVEYOR_01") or {}
        gate = _asset(snapshot, "ACTUATOR_REJECT_01") or {}
        reject_bin = _asset(snapshot, "BIN_REJECT_01") or {}
        answer = (
            f"Machine status: feeder {feeder.get('status', 'unknown')}, camera {camera.get('status', 'unknown')} at "
            f"{camera.get('metrics', {}).get('fps', 'unknown')} FPS, conveyor {conveyor.get('status', 'unknown')}, "
            f"reject gate {gate.get('status', 'unknown')}, and reject bin "
            f"{reject_bin.get('metrics', {}).get('fill_pct', 'unknown')}% full. The AI model is {snapshot.model.status}."
        )
    elif any(term in question for term in ("how many", "count", "accepted", "accept", "passed", "pass", "rejected", "reject")):
        intent = "decision_counts"
        counts = _count_decisions(scope_records)
        scope_name = f"batch {batch_id}" if batch_records else (selected_product or {}).get("name", "the current dashboard scope")
        answer = (
            f"For {scope_name}, {len(scope_records)} inspections are recorded: {counts['pass']} accepted, "
            f"{counts['review']} sent for review, and {counts['reject']} rejected."
        )
    elif any(term in question for term in ("defect", "anomal", "problem", "common")):
        intent = "defect_summary"
        counts = _defect_counts(scope_records)
        answer = (
            f"The main recorded defect types in the current scope are {_format_defects(counts, 5)}. "
            "These labels come from the saved inspection regions and heatmap evidence."
        )
    elif any(term in question for term in ("model", "threshold", "ai ", "algorithm")):
        intent = "model_status"
        answer = (
            f"The active AI model is {snapshot.model.name} ({snapshot.model.model_kind}). Its status is "
            f"{snapshot.model.status} and the current anomaly threshold is {snapshot.model.threshold}. "
            "It compares each captured product image with the learned normal appearance and localizes unusual regions."
        )
    elif any(term in question for term in ("latency", "speed", "fast", "performance", "duration", "fps")):
        intent = "performance"
        durations = [float(item.get("duration_ms", 0)) for item in scope_records]
        average_ms = mean(durations) if durations else 0.0
        camera = _asset(snapshot, "CAMERA_01") or {}
        answer = (
            f"The current scope averages {average_ms:.1f} ms per saved inspection across {len(durations)} records. "
            f"The camera reports {camera.get('metrics', {}).get('fps', 'unknown')} FPS."
        )
    elif any(term in question for term in ("recipe", "product", "capsule", "tablet", "softgel", "blister")):
        intent = "recipe"
        product_name = (selected_product or {}).get("name") or snapshot.recipe.name
        family = (selected_product or {}).get("product_family") or snapshot.recipe.product_family
        sides = (selected_product or {}).get("inspection_sides") or snapshot.recipe.inspection_sides
        channels = (selected_product or {}).get("inspection_channels") or snapshot.recipe.inspection_channels
        answer = (
            f"The selected recipe is {product_name}, a {str(family).replace('_', ' ')} product inspected from "
            f"{sides} side{'s' if sides != 1 else ''} using {', '.join(channels)} imaging."
        )
    elif any(term in question for term in ("happening", "right now", "current status", "live", "doing now")):
        intent = "live_status"
        if snapshot.active_cycle:
            cycle = snapshot.active_cycle
            answer = (
                f"The line is {cycle.status_message.lower()} for image {cycle.batch_position} of {cycle.batch_size} in "
                f"batch {cycle.batch_id}. Progress is {cycle.progress_pct:.0f}% and the current phase is {cycle.phase}."
            )
        else:
            answer = (
                f"The line is {snapshot.inspection_line.status} and waiting for the next product. Batch {snapshot.batch.batch_id} "
                f"is {snapshot.batch.status}; the feeder and conveyor remain ready for automatic inspection."
            )
    elif any(term in question for term in ("latest", "last result", "last inspection", "result")):
        intent = "latest_inspection"
        if selected_inspection:
            answer = (
                f"The displayed inspection is {selected_inspection.get('source_name')}: "
                f"{selected_inspection.get('decision')} with anomaly score {float(selected_inspection.get('score', 0)):.3f} "
                f"against threshold {float(selected_inspection.get('threshold', 0)):.3f}. "
                f"Recorded evidence: {_format_defects(_defect_counts([selected_inspection]))}."
            )
        else:
            answer = "No inspection result has been recorded yet."
    else:
        counts = _count_decisions(scope_records)
        intent = "dashboard_summary"
        answer = (
            f"VisionOps is monitoring {snapshot.recipe.name} on {snapshot.inspection_line.name}. In the current data scope, "
            f"{counts['pass']} items passed, {counts['review']} need review, and {counts['reject']} were rejected. "
            f"The line is {snapshot.inspection_line.status} and the AI model is {snapshot.model.status}."
        )

    return AssistantAnswer(
        answer=answer,
        intent=intent,
        context=context,
        suggested_questions=DEFAULT_SUGGESTIONS,
        generated_at=_now(),
    )
