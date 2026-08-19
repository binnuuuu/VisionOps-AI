from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .config import STORE_FILE, ensure_runtime_dirs


class JsonStore:
    """Tiny durable store for MVP state.

    The app can later move this behind SQLAlchemy/Postgres without changing the
    API contracts.
    """

    def __init__(self, path: Path = STORE_FILE) -> None:
        ensure_runtime_dirs()
        self.path = path
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write_unlocked(self._default_state())

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"products": [], "inspections": [], "model_versions": [], "audit_events": []}

    def read(self) -> dict[str, Any]:
        with self._lock:
            return self._read_unlocked()

    def update(self, mutator) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked()
            mutator(state)
            self._write_unlocked(state)
            return state

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        with self.path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        changed = False
        for key, value in self._default_state().items():
            if key not in state:
                state[key] = value
                changed = True
        for product in state.get("products", []):
            name = str(product.get("name") or "").lower()
            model_kind = str(product.get("model_kind") or "")
            is_capsule = "capsule" in name or model_kind.startswith("patchcore-lite")
            inferred_family = "capsule" if is_capsule else "blister_pack"
            inferred_shape = "capsule" if is_capsule else "blister_pack"
            inferred_mode = "loose_product" if is_capsule else "blister_pack"
            inferred_sides = 6 if is_capsule else 1
            defaults = {
                "active_model_version_id": None,
                "model_kind": None,
                "approved_model_count": 0,
                "product_family": inferred_family,
                "shape": inferred_shape,
                "diameter_mm": None,
                "length_mm": 22.0 if is_capsule else None,
                "width_mm": 8.0 if is_capsule else None,
                "height_mm": 8.0 if is_capsule else None,
                "inspection_sides": inferred_sides,
                "inspection_channels": ["colour", "backlight", "3d"],
                "sorting_mode": "active_sorting_with_verification",
                "inspection_mode": inferred_mode,
            }
            for key, value in defaults.items():
                if key not in product:
                    product[key] = value
                    changed = True
        if changed:
            self._write_unlocked(state)
        return state

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, default=str)
        tmp_path.replace(self.path)
