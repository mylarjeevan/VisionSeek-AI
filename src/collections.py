"""Local collection registry (JSON), no extra infrastructure."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from src.utils import PROJECT_ROOT, infer_source_dir, portable_relpath

REGISTRY_PATH = PROJECT_ROOT / "data" / "collections.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_registry(items: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def upsert_collection(record: dict) -> dict:
    for key in ("source_dir", "metadata_path", "embeddings_path"):
        if record.get(key):
            record[key] = portable_relpath(record[key])
    items = load_registry()
    found = False
    for i, item in enumerate(items):
        if item.get("id") == record.get("id") or (
            item.get("source_dir") and item.get("source_dir") == record.get("source_dir")
        ):
            items[i] = {**item, **record}
            record = items[i]
            found = True
            break
    if not found:
        record.setdefault("id", uuid.uuid4().hex[:10])
        record.setdefault("created_at", _now())
        items.append(record)
    save_registry(items)
    return record


def get_collection(collection_id: str) -> dict | None:
    for item in load_registry():
        if item.get("id") == collection_id:
            return item
    return None


def seed_from_existing_metadata() -> None:
    """Register the bundled COCO sample index if present, using portable relative paths."""
    sample = PROJECT_ROOT / "data" / "processed" / "coco-val-2017-500" / "metadata.json"
    if not sample.exists():
        return
    embeddings = sample.parent / "embeddings.npz"
    try:
        payload = json.loads(sample.read_text(encoding="utf-8"))
        count = len(payload) if isinstance(payload, list) else 0
    except Exception:
        payload = []
        count = 0
    image_directory = infer_source_dir(
        metadata=payload if isinstance(payload, list) else [],
        metadata_path=sample,
        explicit=PROJECT_ROOT / "data" / "raw" / "coco-val-2017-500",
    )
    upsert_collection(
        {
            "id": "coco-val-2017-500",
            "name": "COCO Val 2017 (500)",
            "source_dir": str(image_directory),
            "metadata_path": "data/processed/coco-val-2017-500/metadata.json",
            "embeddings_path": "data/processed/coco-val-2017-500/embeddings.npz",
            "image_count": count,
            "indexed_count": count,
            "has_embeddings": embeddings.exists(),
            "status": "ready",
            "search_available": True,
            "semantic_available": embeddings.exists(),
            "model_used": "yolo11m.pt",
            "created_at": _now(),
        }
    )
