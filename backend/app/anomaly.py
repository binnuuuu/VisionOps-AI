from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ml.visionops_ml.patchcore_lite import PatchCoreLite


MODEL_SIZE = (384, 256)
STD_FLOOR = 0.04
MODEL_FORMAT_VERSION = 2
MIN_DECISION_THRESHOLD = 4.6
REJECT_RATIO = 1.45
MAX_DISPLAY_SCORE = 15.0
FEATURE_LIMIT_DEFAULTS = {
    "dark70": 0.082,
    "dark80": 0.245,
    "edge": 0.046,
    "mean": 0.815,
    "hot": 0.046,
    "p01": 0.360,
}


@dataclass
class InspectionResult:
    score: float
    threshold: float
    decision: str
    heatmap: Image.Image
    defect_regions: list[dict]
    timings_ms: dict[str, float]
    model_cache_hit: bool


@dataclass
class CachedModel:
    path: Path
    mtime: float
    mean: np.ndarray
    std: np.ndarray
    rows: int
    cols: int
    threshold: float
    feature_limits: dict[str, float]
    format_version: int


_MODEL_CACHE: dict[str, CachedModel] = {}
_PATCHCORE_CACHE: dict[str, tuple[float, PatchCoreLite]] = {}


def load_for_model(path: Path, size: tuple[int, int] = MODEL_SIZE) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(image, size, method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    gray = np.asarray(fitted.convert("L"), dtype=np.float32) / 255.0
    return gray, fitted


def _edge_energy(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    return float(np.mean(np.hypot(gx, gy)))


def _central_bbox(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return [
        int(x1 + width * 0.22),
        int(y1 + height * 0.20),
        int(x2 - width * 0.22),
        int(y2 - height * 0.20),
    ]


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, MAX_DISPLAY_SCORE))


def _score_image(
    gray: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rows: int,
    cols: int,
    feature_limits: dict[str, float],
) -> float:
    z_map = np.abs(gray - mean) / np.maximum(std, STD_FLOOR)
    regions = _score_regions(z_map, gray, mean, rows, cols, feature_limits)
    return max((region["score"] for region in regions), default=0.0)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _load_model_cached(model_path: Path) -> tuple[CachedModel, bool, float]:
    started = time.perf_counter()
    resolved = model_path.resolve()
    cache_key = str(resolved)
    mtime = resolved.stat().st_mtime
    cached = _MODEL_CACHE.get(cache_key)
    if cached and cached.mtime == mtime:
        return cached, True, _elapsed_ms(started)

    with np.load(resolved) as model:
        format_version = int(model["format_version"][0]) if "format_version" in model else 1
        threshold = float(model["threshold"][0])
        feature_limits = {
            key: float(model[f"limit_{key}"][0]) if f"limit_{key}" in model else fallback
            for key, fallback in FEATURE_LIMIT_DEFAULTS.items()
        }
        if format_version < MODEL_FORMAT_VERSION:
            threshold = max(threshold, MIN_DECISION_THRESHOLD)
        loaded = CachedModel(
            path=resolved,
            mtime=mtime,
            mean=np.array(model["mean"], dtype=np.float32),
            std=np.array(model["std"], dtype=np.float32),
            rows=int(model["rows"][0]),
            cols=int(model["cols"][0]),
            threshold=threshold,
            feature_limits=feature_limits,
            format_version=format_version,
        )
    _MODEL_CACHE[cache_key] = loaded
    return loaded, False, _elapsed_ms(started)


def _load_patchcore_cached(model_path: Path) -> tuple[PatchCoreLite, bool, float]:
    started = time.perf_counter()
    resolved = model_path.resolve()
    cache_key = str(resolved)
    mtime = resolved.stat().st_mtime
    cached = _PATCHCORE_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1], True, _elapsed_ms(started)

    model = PatchCoreLite.load(resolved)
    _PATCHCORE_CACHE[cache_key] = (mtime, model)
    return model, False, _elapsed_ms(started)


def _grid_regions(rows: int, cols: int, width: int, height: int) -> list[dict]:
    # A conservative inner blister area. It avoids classifying foil edges as a
    # cavity-specific defect in the MVP model.
    left = int(width * 0.08)
    right = int(width * 0.92)
    top = int(height * 0.10)
    bottom = int(height * 0.90)
    cell_w = (right - left) / cols
    cell_h = (bottom - top) / rows
    regions: list[dict] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(left + col * cell_w)
            y1 = int(top + row * cell_h)
            x2 = int(left + (col + 1) * cell_w)
            y2 = int(top + (row + 1) * cell_h)
            regions.append(
                {
                    "cavity": row * cols + col + 1,
                    "row": row + 1,
                    "col": col + 1,
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return regions


def _feature_limits_from_training(
    arrays: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    rows: int,
    cols: int,
) -> dict[str, float]:
    dark70_values: list[float] = []
    dark80_values: list[float] = []
    edge_values: list[float] = []
    mean_values: list[float] = []
    hot_values: list[float] = []
    p01_values: list[float] = []
    regions = _grid_regions(rows, cols, MODEL_SIZE[0], MODEL_SIZE[1])
    for gray in arrays:
        z_map = np.abs(gray - mean) / np.maximum(std, STD_FLOOR)
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            z_crop = z_map[y1:y2, x1:x2]
            cx1, cy1, cx2, cy2 = _central_bbox(region["bbox"])
            central = gray[cy1:cy2, cx1:cx2]
            dark70_values.append(float(np.mean(central < 0.70)))
            dark80_values.append(float(np.mean(central < 0.80)))
            edge_values.append(_edge_energy(central))
            mean_values.append(float(central.mean()))
            hot_values.append(float(np.mean(z_crop > 2.7)))
            p01_values.append(float(np.percentile(central, 1)))

    return {
        "dark70": max(dark70_values, default=FEATURE_LIMIT_DEFAULTS["dark70"]) + 0.012,
        "dark80": max(dark80_values, default=FEATURE_LIMIT_DEFAULTS["dark80"]) + 0.035,
        "edge": max(edge_values, default=FEATURE_LIMIT_DEFAULTS["edge"]) + 0.006,
        "mean": min(mean_values, default=FEATURE_LIMIT_DEFAULTS["mean"]) - 0.070,
        "hot": max(hot_values, default=FEATURE_LIMIT_DEFAULTS["hot"]) + 0.014,
        "p01": min(p01_values, default=FEATURE_LIMIT_DEFAULTS["p01"]) - 0.100,
    }


def train_normal_model(
    image_paths: list[Path],
    model_path: Path,
    *,
    rows: int,
    cols: int,
) -> dict:
    if len(image_paths) < 3:
        raise ValueError("At least 3 good samples are required for the MVP normal model.")

    arrays = [load_for_model(path)[0] for path in image_paths]
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    feature_limits = _feature_limits_from_training(arrays, mean, std, rows, cols)
    training_scores = [_score_image(gray, mean, std, rows, cols, feature_limits) for gray in arrays]
    threshold = max(
        float(np.percentile(training_scores, 95)) * 1.25,
        max(training_scores) * 1.18,
        MIN_DECISION_THRESHOLD,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.array([MODEL_FORMAT_VERSION], dtype=np.int16),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "rows": np.array([rows], dtype=np.int16),
        "cols": np.array([cols], dtype=np.int16),
        "threshold": np.array([threshold], dtype=np.float32),
    }
    payload.update({f"limit_{key}": np.array([value], dtype=np.float32) for key, value in feature_limits.items()})
    np.savez_compressed(model_path, **payload)

    return {
        "threshold": threshold,
        "training_scores": [round(float(score), 4) for score in training_scores],
        "sample_count": len(image_paths),
    }


def _score_regions(
    z_map: np.ndarray,
    gray: np.ndarray,
    mean: np.ndarray,
    rows: int,
    cols: int,
    feature_limits: dict[str, float],
) -> list[dict]:
    scored_regions: list[dict] = []
    for region in _grid_regions(rows, cols, z_map.shape[1], z_map.shape[0]):
        x1, y1, x2, y2 = region["bbox"]
        z_crop = z_map[y1:y2, x1:x2]
        gray_crop = gray[y1:y2, x1:x2]
        mean_crop = mean[y1:y2, x1:x2]
        if z_crop.size == 0:
            continue

        cx1, cy1, cx2, cy2 = _central_bbox(region["bbox"])
        central = gray[cy1:cy2, cx1:cx2]
        central_mean = float(central.mean())
        central_dark70 = float(np.mean(central < 0.70))
        central_dark80 = float(np.mean(central < 0.80))
        central_p01 = float(np.percentile(central, 1))
        central_edge = _edge_energy(central)
        z98 = float(np.percentile(z_crop, 98.5))
        hot_ratio = float(np.mean(z_crop > 2.7))
        mean_delta = float(gray_crop.mean() - mean_crop.mean())
        p05_delta = float(np.percentile(gray_crop, 5) - np.percentile(mean_crop, 5))
        p95_delta = float(np.percentile(gray_crop, 95) - np.percentile(mean_crop, 95))

        area_score = z98 * min(1.25, hot_ratio / max(feature_limits["hot"], 0.035))
        missing_score = (
            max(0.0, feature_limits["mean"] - central_mean) / 0.035 * 5.0
            + max(0.0, central_dark80 - feature_limits["dark80"]) / 0.050 * 3.0
            + max(0.0, hot_ratio - feature_limits["hot"]) / 0.040 * 2.0
        )
        spot_score = (
            max(0.0, central_dark70 - feature_limits["dark70"]) / 0.011 * 3.0
            + max(0.0, feature_limits["p01"] - central_p01) / 0.060 * 3.0
            + max(0.0, central_edge - feature_limits["edge"]) / 0.006 * 2.2
            + max(0.0, hot_ratio - feature_limits["hot"]) / 0.020 * 1.5
        )
        scratch_score = (
            max(0.0, central_edge - feature_limits["edge"]) / 0.007 * 3.0
            + max(0.0, hot_ratio - feature_limits["hot"]) / 0.030 * 1.5
        )
        raw_score = max(area_score, missing_score, spot_score, scratch_score)
        scored_regions.append(
            {
                **region,
                "central_bbox": [cx1, cy1, cx2, cy2],
                "score": round(_clip_score(raw_score), 4),
                "raw_score": raw_score,
                "features": {
                    "z98": z98,
                    "hot_ratio": hot_ratio,
                    "mean_delta": mean_delta,
                    "p05_delta": p05_delta,
                    "p95_delta": p95_delta,
                    "central_mean": central_mean,
                    "central_dark70": central_dark70,
                    "central_dark80": central_dark80,
                    "central_p01": central_p01,
                    "central_edge": central_edge,
                    "area_score": area_score,
                    "missing_score": missing_score,
                    "spot_score": spot_score,
                    "scratch_score": scratch_score,
                    "limit_dark70": feature_limits["dark70"],
                    "limit_dark80": feature_limits["dark80"],
                    "limit_edge": feature_limits["edge"],
                    "limit_mean": feature_limits["mean"],
                    "limit_hot": feature_limits["hot"],
                    "limit_p01": feature_limits["p01"],
                },
            }
        )
    return sorted(scored_regions, key=lambda item: item["score"], reverse=True)


def inspect_with_patchcore_lite_model(model_path: Path, image_path: Path) -> InspectionResult:
    timings: dict[str, float] = {}
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, cache_hit, timings["model_load"] = _load_patchcore_cached(model_path)

    started = time.perf_counter()
    source = Image.open(image_path).convert("RGB")
    fitted = ImageOps.fit(
        source,
        (model.config.image_width, model.config.image_height),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    timings["preprocess"] = _elapsed_ms(started)

    started = time.perf_counter()
    prediction = model.predict(image_path)
    timings["inference"] = _elapsed_ms(started)

    score = float(prediction.score)
    threshold = float(prediction.threshold)
    started = time.perf_counter()
    artifact_regions = _find_capsule_artifact_regions(fitted, score, threshold)
    if artifact_regions:
        score = max(score, threshold * 1.2)

    if score >= threshold * 1.18:
        decision = "reject"
    elif score >= threshold:
        decision = "review"
    else:
        decision = "pass"

    defect_regions = artifact_regions if artifact_regions else ([] if decision == "pass" else _patchcore_regions(prediction.patch_scores, fitted, score, threshold))
    timings["localization"] = _elapsed_ms(started)

    started = time.perf_counter()
    heatmap = _make_patchcore_heatmap(fitted, prediction.patch_scores, defect_regions)
    timings["heatmap"] = _elapsed_ms(started)
    return InspectionResult(
        score=round(score, 4),
        threshold=round(threshold, 4),
        decision=decision,
        heatmap=heatmap,
        defect_regions=defect_regions,
        timings_ms=timings,
        model_cache_hit=cache_hit,
    )


def _component_from_seed(mask: np.ndarray, seed_y: int, seed_x: int) -> list[tuple[int, int]]:
    stack = [(int(seed_y), int(seed_x))]
    visited: set[tuple[int, int]] = set()
    component: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        if (y, x) in visited:
            continue
        visited.add((y, x))
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1] or not bool(mask[y, x]):
            continue
        component.append((y, x))
        for next_y in range(y - 1, y + 2):
            for next_x in range(x - 1, x + 2):
                if (next_y, next_x) != (y, x):
                    stack.append((next_y, next_x))
    return component


def _binary_components(mask: np.ndarray) -> list[dict]:
    if mask.size == 0:
        return []

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict] = []
    height, width = mask.shape
    for seed_y, seed_x in np.argwhere(mask):
        seed = (int(seed_y), int(seed_x))
        if visited[seed]:
            continue
        stack = [seed]
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            if y < 0 or x < 0 or y >= height or x >= width or visited[y, x] or not bool(mask[y, x]):
                continue
            visited[y, x] = True
            pixels.append((y, x))
            stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
        if not pixels:
            continue
        ys = np.asarray([pixel[0] for pixel in pixels], dtype=np.int32)
        xs = np.asarray([pixel[1] for pixel in pixels], dtype=np.int32)
        comp_w = int(xs.max() - xs.min() + 1)
        comp_h = int(ys.max() - ys.min() + 1)
        components.append(
            {
                "area": len(pixels),
                "area_ratio": len(pixels) / float(mask.size),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                "width": comp_w,
                "height": comp_h,
                "long_side": max(comp_w, comp_h),
                "aspect": max(comp_w, comp_h) / max(float(min(comp_w, comp_h)), 1.0),
            }
        )
    return sorted(components, key=lambda item: item["area"], reverse=True)


def _classify_capsule_defect(rgb: Image.Image, bbox: list[int]) -> tuple[str, float]:
    x1, y1, x2, y2 = bbox
    crop = np.asarray(rgb.crop((x1, y1, x2, y2)), dtype=np.float32) / 255.0
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return "capsule_surface_deviation", 0.45

    red = crop[..., 0]
    green = crop[..., 1]
    blue = crop[..., 2]
    maximum = np.maximum.reduce([red, green, blue])
    minimum = np.minimum.reduce([red, green, blue])
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    gray = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)

    high_edge = edge > np.percentile(edge, 82)
    dark_limit = max(0.22, float(np.percentile(gray, 12)))
    dark_mask = gray < dark_limit
    fiber_mask = dark_mask & (saturation < 0.45) & high_edge
    fiber_components = _binary_components(fiber_mask)
    best_fiber = fiber_components[0] if fiber_components else None

    brown_mask = (
        ((red > green * 0.82) & (green > blue * 0.72) & (saturation > 0.16) & (gray < 0.62))
        | ((gray < 0.26) & (saturation > 0.10))
    )
    brown_ratio = float(np.mean(brown_mask))
    dark_ratio = float(np.mean(gray < 0.24))
    brown_components = _binary_components(brown_mask)
    largest_brown = float(brown_components[0]["area_ratio"]) if brown_components else 0.0

    line_mask = (gray < 0.48) & (saturation < 0.62)
    line_components = _binary_components(line_mask)
    best_line = next(
        (
            component
            for component in line_components
            if component["aspect"] >= 2.65
            and component["area_ratio"] <= 0.09
            and component["long_side"] >= min(crop.shape[0], crop.shape[1]) * 0.45
        ),
        None,
    )
    very_dark_y, very_dark_x = np.where(gray < 0.12)
    very_dark_trace = None
    if len(very_dark_x):
        trace_w = int(very_dark_x.max() - very_dark_x.min() + 1)
        trace_h = int(very_dark_y.max() - very_dark_y.min() + 1)
        very_dark_trace = {
            "area_ratio": len(very_dark_x) / float(gray.size),
            "long_side": max(trace_w, trace_h),
            "aspect": max(trace_w, trace_h) / max(float(min(trace_w, trace_h)), 1.0),
        }
    if (
        best_fiber
        and best_fiber["aspect"] >= 4.0
        and best_fiber["area_ratio"] <= 0.09
        and best_fiber["long_side"] >= min(crop.shape[0], crop.shape[1]) * 0.20
    ) or (best_line and brown_ratio < 0.24) or (
        very_dark_trace
        and 0.015 <= very_dark_trace["area_ratio"] <= 0.12
        and very_dark_trace["aspect"] >= 2.2
        and very_dark_trace["long_side"] >= min(crop.shape[0], crop.shape[1]) * 0.62
    ):
        aspect = float((very_dark_trace or best_line or best_fiber)["aspect"])
        confidence = 0.62 + min(0.28, (aspect - 3.0) * 0.045)
        return "hair_or_fiber_contamination", round(float(confidence), 3)

    light_mask = (gray > 0.68) & (saturation < 0.48)
    light_ratio = float(np.mean(light_mask))
    light_components = _binary_components(light_mask)
    largest_light = float(light_components[0]["area_ratio"]) if light_components else 0.0
    if light_ratio >= 0.18 and largest_light >= 0.08 and brown_ratio < 0.24 and dark_ratio < 0.18:
        confidence = 0.54 + min(0.30, (light_ratio * 0.85) + (largest_light * 1.5))
        return "bubble_or_fill_void", round(float(confidence), 3)

    if brown_ratio >= 0.16 or largest_brown >= 0.10 or dark_ratio >= 0.20:
        confidence = 0.58 + min(0.34, (brown_ratio * 0.75) + (largest_brown * 0.9) + (dark_ratio * 0.45))
        return "broken_capsule_or_leak", round(float(confidence), 3)

    spot_mask = ((gray < 0.34) & (saturation > 0.18)) | ((gray > 0.74) & (saturation < 0.36))
    spot_components = _binary_components(spot_mask)
    if spot_components and spot_components[0]["area_ratio"] <= 0.16:
        confidence = 0.50 + min(0.28, spot_components[0]["area_ratio"] * 1.4)
        return "surface_particle_or_stain", round(float(confidence), 3)

    return "capsule_surface_deviation", 0.5


def _find_capsule_artifact_regions(rgb: Image.Image, score: float, threshold: float) -> list[dict]:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    maximum = np.maximum.reduce([red, green, blue])
    minimum = np.minimum.reduce([red, green, blue])
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    gray = (0.299 * red) + (0.587 * green) + (0.114 * blue)

    capsule_mask = (
        (green > red * 1.03)
        & (green > blue * 1.10)
        & (saturation > 0.22)
        & (gray > 0.24)
        & (gray < 0.92)
    )
    blurred = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)),
        dtype=np.float32,
    ) / 255.0
    contrast = gray - blurred
    bubble_mask = (
        capsule_mask
        & (contrast > 0.045)
        & (gray > 0.43)
        & (gray < 0.90)
        & (saturation > 0.50)
        & (green > 0.55)
    )

    candidates: list[dict] = []
    height, width = gray.shape
    for component in _binary_components(bubble_mask):
        area = int(component["area"])
        comp_w = int(component["width"])
        comp_h = int(component["height"])
        x1, y1, x2, y2 = component["bbox"]
        if not (8 <= area <= 70 and 3 <= comp_w <= 12 and 3 <= comp_h <= 12 and component["aspect"] <= 2.60):
            continue

        crop_mask = bubble_mask[y1:y2, x1:x2]
        if not crop_mask.any():
            continue
        fill_ratio = area / float(comp_w * comp_h)
        local_contrast = contrast[y1:y2, x1:x2][crop_mask]
        local_gray = gray[y1:y2, x1:x2][crop_mask]
        local_saturation = saturation[y1:y2, x1:x2][crop_mask]
        local_green = green[y1:y2, x1:x2][crop_mask]

        pad = 8
        ring_x1 = max(0, x1 - pad)
        ring_y1 = max(0, y1 - pad)
        ring_x2 = min(width, x2 + pad)
        ring_y2 = min(height, y2 + pad)
        ring = np.ones((ring_y2 - ring_y1, ring_x2 - ring_x1), dtype=bool)
        ring[y1 - ring_y1 : y2 - ring_y1, x1 - ring_x1 : x2 - ring_x1] = False
        capsule_context = float(capsule_mask[ring_y1:ring_y2, ring_x1:ring_x2][ring].mean()) if ring.any() else 0.0
        mean_contrast = float(local_contrast.mean())
        max_gray = float(local_gray.max())
        mean_saturation = float(local_saturation.mean())
        mean_green = float(local_green.mean())
        compact_ring = (
            8 <= area <= 70
            and 4 <= comp_w <= 12
            and 4 <= comp_h <= 12
            and component["aspect"] <= 1.45
            and fill_ratio >= 0.38
            and mean_contrast >= 0.10
            and 0.72 <= max_gray <= 0.91
            and mean_saturation >= 0.58
            and mean_green >= 0.72
            and capsule_context >= 0.60
        )
        # A bubble crossing a strong specular streak appears as a short partial
        # ring instead of a compact circle. These stricter context/color limits
        # keep that case separate from normal elongated highlights.
        partial_ring = (
            12 <= area <= 45
            and 4 <= comp_w <= 12
            and 3 <= comp_h <= 8
            and 1.45 <= component["aspect"] <= 2.60
            and fill_ratio >= 0.48
            and mean_contrast >= 0.082
            and 0.72 <= max_gray <= 0.90
            and mean_saturation >= 0.68
            and mean_green >= 0.80
            and capsule_context >= 0.72
        )
        if not compact_ring and not partial_ring:
            continue

        bubble_strength = mean_contrast + (capsule_context * 0.08) + (fill_ratio * 0.05)
        candidates.append(
            {
                **component,
                "bubble_strength": bubble_strength,
                "capsule_context": capsule_context,
                "mean_contrast": mean_contrast,
                "partial_ring": partial_ring,
            }
        )

    if not candidates:
        return []

    best = max(candidates, key=lambda item: item["bubble_strength"])
    x1, y1, x2, y2 = best["bbox"]
    pad_x = 12
    pad_y = 12
    bbox = [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]
    confidence_floor = 0.76 if best.get("partial_ring") else 0.70
    confidence = float(
        np.clip(
            confidence_floor + ((best["mean_contrast"] - 0.082) * 1.8) + ((best["capsule_context"] - 0.60) * 0.25),
            confidence_floor,
            0.94,
        )
    )
    return [
        {
            "cavity": 0,
            "row": 0,
            "col": 0,
            "score": round(max(score, threshold * 1.2), 4),
            "bbox": bbox,
            "defect_type": "bubble_or_fill_void",
            "severity": "high",
            "confidence": round(confidence, 3),
            "area_pct": round(float(best["area"]) / float(width * height) * 100.0, 3),
        }
    ]


def _patchcore_regions(score_map: np.ndarray, rgb: Image.Image, score: float, threshold: float) -> list[dict]:
    width, height = rgb.size
    normalized = score_map.astype(np.float32)
    cutoff = max(float(np.percentile(normalized, 99.2)), threshold)
    mask = normalized >= cutoff
    max_y, max_x = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    mask[max_y, max_x] = True

    component = _component_from_seed(mask, int(max_y), int(max_x))

    ys = np.asarray([item[0] for item in component], dtype=np.int32)
    xs = np.asarray([item[1] for item in component], dtype=np.int32)
    grid_h, grid_w = normalized.shape
    pad_x = max(14, int((width / grid_w) * 2.4))
    pad_y = max(14, int((height / grid_h) * 2.4))
    x1 = max(int(xs.min() / grid_w * width) - pad_x, 0)
    y1 = max(int(ys.min() / grid_h * height) - pad_y, 0)
    x2 = min(int((xs.max() + 1) / grid_w * width) + pad_x, width)
    y2 = min(int((ys.max() + 1) / grid_h * height) + pad_y, height)
    ratio = score / max(threshold, 1e-6)
    defect_type, subtype_confidence = _classify_capsule_defect(rgb, [x1, y1, x2, y2])
    severity = "critical" if ratio >= 1.18 or defect_type == "broken_capsule_or_leak" else "high"
    anomaly_confidence = float(np.clip(0.45 + ((ratio - 1.0) * 0.55), 0.45, 0.98))
    confidence = max(anomaly_confidence, subtype_confidence)
    area_pct = float((len(component) / normalized.size) * 100)
    return [
        {
            "cavity": 0,
            "row": 0,
            "col": 0,
            "score": round(score, 4),
            "bbox": [x1, y1, x2, y2],
            "defect_type": defect_type,
            "severity": severity,
            "confidence": round(confidence, 3),
            "area_pct": round(area_pct, 2),
        }
    ]


def _make_patchcore_heatmap(rgb: Image.Image, score_map: np.ndarray, defect_regions: list[dict]) -> Image.Image:
    if not defect_regions:
        return rgb

    normalized = score_map.astype(np.float32)
    normalized = normalized - float(normalized.min())
    normalized = normalized / max(float(normalized.max()), 1e-6)
    normalized = np.clip((normalized - 0.68) / 0.32, 0.0, 1.0)
    heat = Image.fromarray((normalized * 255).astype(np.uint8))
    heat = heat.resize(rgb.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=1.2))
    alpha = np.asarray(heat, dtype=np.uint8)
    overlay = np.zeros((rgb.height, rgb.width, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = 154
    overlay[..., 2] = 44
    overlay[..., 3] = np.clip(alpha.astype(np.float32) * 0.32, 0, 92).astype(np.uint8)
    output = Image.alpha_composite(rgb.convert("RGBA"), Image.fromarray(overlay)).convert("RGB")
    draw = ImageDraw.Draw(output)
    for region in defect_regions:
        x1, y1, x2, y2 = region["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(230, 104, 34), width=2)
        label = _short_defect_label(region.get("defect_type", "capsule_surface_deviation"))
        text_y = y1 - 15 if y1 >= 18 else min(y2 + 4, output.height - 14)
        text_bbox = draw.textbbox((x1 + 4, text_y), label)
        draw.rectangle(
            [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2],
            fill=(20, 28, 27),
        )
        draw.text((x1 + 4, text_y), label, fill=(255, 255, 255))
    return output


def inspect_with_model(model_path: Path, image_path: Path) -> InspectionResult:
    timings: dict[str, float] = {}
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, cache_hit, timings["model_load"] = _load_model_cached(model_path)

    started = time.perf_counter()
    gray, rgb = load_for_model(image_path)
    timings["preprocess"] = _elapsed_ms(started)

    started = time.perf_counter()
    z_map = np.abs(gray - model.mean) / np.maximum(model.std, STD_FLOOR)
    scored_regions = _score_regions(z_map, gray, model.mean, model.rows, model.cols, model.feature_limits)
    score = max((region["score"] for region in scored_regions), default=0.0)
    timings["inference"] = _elapsed_ms(started)

    if score >= model.threshold * REJECT_RATIO:
        decision = "reject"
    elif score >= model.threshold:
        decision = "review"
    else:
        decision = "pass"

    started = time.perf_counter()
    defect_regions = [] if decision == "pass" else _find_defect_regions(scored_regions, z_map, model.threshold)
    timings["localization"] = _elapsed_ms(started)

    started = time.perf_counter()
    heatmap = _make_heatmap(rgb, z_map, model.threshold, defect_regions)
    timings["heatmap"] = _elapsed_ms(started)
    return InspectionResult(
        score=round(score, 4),
        threshold=round(model.threshold, 4),
        decision=decision,
        heatmap=heatmap,
        defect_regions=defect_regions,
        timings_ms=timings,
        model_cache_hit=cache_hit,
    )


def _make_heatmap(rgb: Image.Image, z_map: np.ndarray, threshold: float, defect_regions: list[dict]) -> Image.Image:
    if not defect_regions:
        return rgb

    normalized = np.clip((z_map - 3.0) / 5.0, 0, 1)
    alpha = (normalized * 82).astype(np.uint8)
    highlight = np.zeros((*z_map.shape, 4), dtype=np.uint8)
    highlight[..., 0] = 255
    highlight[..., 1] = 154
    highlight[..., 2] = 44
    highlight[..., 3] = alpha
    overlay = Image.fromarray(highlight).filter(ImageFilter.GaussianBlur(radius=0.8))
    composed = Image.alpha_composite(rgb.convert("RGBA"), overlay)
    output = composed.convert("RGB")
    draw = ImageDraw.Draw(output)
    for region in defect_regions:
        x1, y1, x2, y2 = region["bbox"]
        severity = region.get("severity", "moderate")
        color = {
            "critical": (190, 32, 25),
            "high": (230, 104, 34),
            "moderate": (244, 176, 49),
        }.get(severity, (244, 176, 49))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"C{region['cavity']} {_short_defect_label(region.get('defect_type', 'unclassified_deviation'))}"
        text_bbox = draw.textbbox((x1 + 4, y1 + 4), label)
        draw.rectangle(
            [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2],
            fill=(20, 28, 27),
        )
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255))
    return output


def _short_defect_label(defect_type: str) -> str:
    labels = {
        "missing_or_empty_cavity": "missing",
        "foreign_particle_or_stain": "particle",
        "foil_tear_or_surface_damage": "foil damage",
        "cracked_or_broken_tablet": "crack",
        "fiber_or_hair_contamination": "fiber",
        "shape_or_fill_mismatch": "fill mismatch",
        "broken_capsule_or_leak": "broken/leak",
        "hair_or_fiber_contamination": "hair/fiber",
        "bubble_or_fill_void": "bubble/void",
        "surface_particle_or_stain": "particle/stain",
        "capsule_surface_deviation": "surface",
        "unclassified_deviation": "deviation",
        "visual_anomaly": "deviation",
    }
    return labels.get(defect_type, defect_type.replace("_", " "))


def _classify_defect(features: dict, score: float, threshold: float) -> tuple[str, str, float]:
    central_mean = float(features["central_mean"])
    central_dark70 = float(features["central_dark70"])
    central_dark80 = float(features["central_dark80"])
    central_p01 = float(features["central_p01"])
    central_edge = float(features["central_edge"])
    hot_ratio = float(features["hot_ratio"])
    p05_delta = float(features["p05_delta"])
    p95_delta = float(features["p95_delta"])
    mean_delta = float(features["mean_delta"])
    dark70_limit = float(features["limit_dark70"])
    dark80_limit = float(features["limit_dark80"])
    edge_limit = float(features["limit_edge"])
    mean_limit = float(features["limit_mean"])
    hot_limit = float(features["limit_hot"])
    p01_limit = float(features["limit_p01"])

    if central_mean < mean_limit or central_dark80 > dark80_limit + 0.24:
        defect_type = "missing_or_empty_cavity"
    elif central_edge > edge_limit + 0.026 and central_dark70 > dark70_limit + 0.032:
        defect_type = "cracked_or_broken_tablet"
    elif central_edge > edge_limit + 0.014 and p05_delta < -0.080:
        defect_type = "fiber_or_hair_contamination"
    elif central_dark70 > dark70_limit + 0.010 or central_p01 < p01_limit:
        defect_type = "foreign_particle_or_stain"
    elif p05_delta < -0.120 or hot_ratio > hot_limit + 0.034:
        defect_type = "foil_tear_or_surface_damage"
    elif p95_delta < -0.050 or abs(mean_delta) > 0.025:
        defect_type = "shape_or_fill_mismatch"
    else:
        defect_type = "unclassified_deviation"

    ratio = score / max(threshold, 0.001)
    if ratio >= REJECT_RATIO:
        severity = "critical"
    elif ratio >= 1.0:
        severity = "high"
    else:
        severity = "moderate"
    confidence = float(np.clip(0.42 + ((ratio - 0.85) * 0.52) + min(score, 12) / 38, 0.35, 0.99))
    return defect_type, severity, round(confidence, 3)


def _find_defect_regions(scored_regions: list[dict], z_map: np.ndarray, threshold: float) -> list[dict]:
    regions = []
    for region in scored_regions:
        if region["score"] < threshold:
            continue
        x1, y1, x2, y2 = region["bbox"]
        crop = z_map[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mask = crop >= max(3.0, float(np.percentile(crop, 98.0)))
        area_pct = float(mask.mean() * 100)
        if mask.any():
            ys, xs = np.where(mask)
            pad = 5
            bx1 = max(x1 + int(xs.min()) - pad, x1)
            by1 = max(y1 + int(ys.min()) - pad, y1)
            bx2 = min(x1 + int(xs.max()) + pad, x2)
            by2 = min(y1 + int(ys.max()) + pad, y2)
        else:
            bx1, by1, bx2, by2 = region["central_bbox"]
        defect_type, severity, confidence = _classify_defect(region["features"], region["score"], threshold)
        regions.append(
            {
                "cavity": region["cavity"],
                "row": region["row"],
                "col": region["col"],
                "bbox": [bx1, by1, bx2, by2],
                "score": round(float(region["score"]), 4),
                "defect_type": defect_type,
                "severity": severity,
                "confidence": confidence,
                "area_pct": round(area_pct, 2),
            }
        )
    regions.sort(key=lambda item: item["score"], reverse=True)
    return regions[:5]


def draw_grid_preview(image_path: Path, output_path: Path, rows: int, cols: int) -> None:
    _, rgb = load_for_model(image_path)
    draw = ImageDraw.Draw(rgb)
    for region in _grid_regions(rows, cols, rgb.width, rgb.height):
        draw.rectangle(region["bbox"], outline=(27, 116, 142), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_path)
