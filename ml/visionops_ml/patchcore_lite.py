from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class PatchCoreLiteConfig:
    image_width: int = 384
    image_height: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory_patches: int = 12000
    threshold_quantile: float = 99.0
    threshold_std_factor: float = 3.0
    min_threshold: float = 0.01
    seed: int = 7


@dataclass(frozen=True)
class PatchCorePrediction:
    score: float
    threshold: float
    is_anomaly: bool
    patch_scores: np.ndarray
    grid_shape: tuple[int, int]


def _load_image(path: Path, config: PatchCoreLiteConfig) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(
        image,
        (config.image_width, config.image_height),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted, dtype=np.float32) / 255.0


def _feature_map(image: np.ndarray) -> np.ndarray:
    gray = (0.299 * image[..., 0]) + (0.587 * image[..., 1]) + (0.114 * image[..., 2])
    gy, gx = np.gradient(gray)
    edges = np.sqrt((gx * gx) + (gy * gy))
    return np.dstack([image, gray, edges])


def extract_patch_features(image_path: Path, config: PatchCoreLiteConfig) -> tuple[np.ndarray, tuple[int, int]]:
    image = _feature_map(_load_image(image_path, config))
    height, width, channels = image.shape
    features: list[np.ndarray] = []
    grid_h = 0
    grid_w = 0
    for y in range(0, height - config.patch_size + 1, config.stride):
        grid_w = 0
        for x in range(0, width - config.patch_size + 1, config.stride):
            patch = image[y : y + config.patch_size, x : x + config.patch_size, :]
            mean = patch.mean(axis=(0, 1))
            std = patch.std(axis=(0, 1))
            low = np.percentile(patch, 10, axis=(0, 1))
            high = np.percentile(patch, 90, axis=(0, 1))
            features.append(np.concatenate([mean, std, low, high], axis=0))
            grid_w += 1
        grid_h += 1
    return np.asarray(features, dtype=np.float32), (grid_h, grid_w)


def _nearest_distances(features: np.ndarray, memory_bank: np.ndarray, batch_size: int = 512) -> np.ndarray:
    distances: list[np.ndarray] = []
    features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=50.0, neginf=-50.0)
    memory_bank = np.nan_to_num(memory_bank.astype(np.float32), nan=0.0, posinf=50.0, neginf=-50.0)
    memory_sq = np.sum(memory_bank * memory_bank, axis=1)
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        batch_sq = np.sum(batch * batch, axis=1, keepdims=True)
        dot = np.einsum("ij,kj->ik", batch, memory_bank, optimize=True)
        sq = batch_sq + memory_sq[None, :] - (2.0 * dot)
        distances.append(np.sqrt(np.maximum(np.min(sq, axis=1), 0.0)))
    return np.concatenate(distances, axis=0)


class PatchCoreLite:
    def __init__(
        self,
        config: PatchCoreLiteConfig | None = None,
        *,
        memory_bank: np.ndarray | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> None:
        self.config = config or PatchCoreLiteConfig()
        self.memory_bank = memory_bank
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.threshold = threshold

    def fit(self, image_paths: list[Path]) -> dict:
        if len(image_paths) < 3:
            raise ValueError("At least 3 normal images are required.")

        calibration_count = 0
        if len(image_paths) >= 8:
            calibration_count = max(2, min(64, int(round(len(image_paths) * 0.1))))
        memory_paths = image_paths[:-calibration_count] if calibration_count else image_paths
        calibration_paths = image_paths[-calibration_count:] if calibration_count else image_paths

        memory_feature_sets = [extract_patch_features(path, self.config)[0] for path in memory_paths]
        memory_features = np.vstack(memory_feature_sets)
        self.feature_mean = memory_features.mean(axis=0)
        self.feature_std = np.maximum(memory_features.std(axis=0), 1e-3)
        normalized = np.clip((memory_features - self.feature_mean) / self.feature_std, -50.0, 50.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=50.0, neginf=-50.0)

        rng = np.random.default_rng(self.config.seed)
        if len(normalized) > self.config.max_memory_patches:
            indices = rng.choice(len(normalized), size=self.config.max_memory_patches, replace=False)
            memory = normalized[indices]
        else:
            memory = normalized
        self.memory_bank = memory.astype(np.float32)

        training_scores = []
        for path in calibration_paths:
            features, _ = extract_patch_features(path, self.config)
            prediction = self._predict_features(features, threshold=0.0)
            training_scores.append(prediction.score)
        self.threshold = float(
            max(
                np.percentile(training_scores, self.config.threshold_quantile),
                float(np.mean(training_scores) + (self.config.threshold_std_factor * np.std(training_scores))),
                self.config.min_threshold,
            )
        )

        return {
            "normal_images": len(image_paths),
            "memory_images": len(memory_paths),
            "calibration_images": len(calibration_paths),
            "memory_patches": int(len(self.memory_bank)),
            "training_scores": [round(float(score), 6) for score in training_scores],
            "threshold": round(float(self.threshold), 6),
        }

    def predict(self, image_path: Path) -> PatchCorePrediction:
        features, grid_shape = extract_patch_features(image_path, self.config)
        return self._predict_features(features, grid_shape=grid_shape, threshold=self._require_threshold())

    def _predict_features(
        self,
        features: np.ndarray,
        *,
        grid_shape: tuple[int, int] | None = None,
        threshold: float,
    ) -> PatchCorePrediction:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None:
            raise ValueError("Model is not fitted.")
        normalized = np.clip((features - self.feature_mean) / self.feature_std, -50.0, 50.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=50.0, neginf=-50.0)
        patch_scores = _nearest_distances(normalized.astype(np.float32), self.memory_bank)
        score = float(np.percentile(patch_scores, 99))
        if grid_shape is None:
            grid_shape = (1, len(patch_scores))
        return PatchCorePrediction(
            score=round(score, 6),
            threshold=round(float(threshold), 6),
            is_anomaly=score >= threshold,
            patch_scores=patch_scores.reshape(grid_shape),
            grid_shape=grid_shape,
        )

    def save(self, path: Path, metadata: dict | None = None) -> None:
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise ValueError("Model is not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            threshold=np.asarray([self.threshold], dtype=np.float32),
            config=np.asarray([json.dumps(self.config.__dict__)]),
            metadata=np.asarray([json.dumps(metadata or {})]),
        )

    @classmethod
    def load(cls, path: Path) -> "PatchCoreLite":
        model = np.load(path, allow_pickle=False)
        config = PatchCoreLiteConfig(**json.loads(str(model["config"][0])))
        return cls(
            config,
            memory_bank=model["memory_bank"],
            feature_mean=model["feature_mean"],
            feature_std=model["feature_std"],
            threshold=float(model["threshold"][0]),
        )

    def heatmap(self, image_path: Path, prediction: PatchCorePrediction, output_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        fitted = ImageOps.fit(
            image,
            (self.config.image_width, self.config.image_height),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        score_map = prediction.patch_scores
        score_map = score_map - score_map.min()
        score_map = score_map / max(float(score_map.max()), 1e-6)
        heat = Image.fromarray((score_map * 255).astype(np.uint8))
        heat = heat.resize(fitted.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=2))
        alpha = np.asarray(heat, dtype=np.uint8)
        overlay = np.zeros((fitted.height, fitted.width, 4), dtype=np.uint8)
        overlay[..., 0] = 235
        overlay[..., 1] = 67
        overlay[..., 2] = 48
        overlay[..., 3] = np.clip(alpha.astype(np.float32) * 0.68, 0, 180).astype(np.uint8)
        composed = Image.alpha_composite(fitted.convert("RGBA"), Image.fromarray(overlay))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.convert("RGB").save(output_path)

    def _require_threshold(self) -> float:
        if self.threshold is None:
            raise ValueError("Model threshold is missing.")
        return self.threshold
