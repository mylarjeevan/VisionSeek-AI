"""Normalize and inspect detection metadata (old and new shapes)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any


def detection_label(det: dict) -> str:
    return str(det.get("class") or det.get("label") or "").strip().lower()


def detection_confidence(det: dict) -> float:
    try:
        return float(det.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def detections_of(item: dict) -> list[dict]:
    rows = item.get("detections") or item.get("objects") or []
    return rows if isinstance(rows, list) else []


def class_counts_of(item: dict) -> dict[str, int]:
    counts = item.get("class_counts") or item.get("object_counts") or {}
    if counts:
        return {str(k).lower(): int(v) for k, v in counts.items()}
    tallies: dict[str, int] = {}
    for det in detections_of(item):
        label = detection_label(det)
        if label:
            tallies[label] = tallies.get(label, 0) + 1
    return tallies


def classes_present(item: dict, min_conf: float | None = None, max_conf: float | None = None) -> set[str]:
    present: set[str] = set()
    for det in detections_of(item):
        conf = detection_confidence(det)
        if min_conf is not None and conf < min_conf:
            continue
        if max_conf is not None and conf > max_conf:
            continue
        label = detection_label(det)
        if label:
            present.add(label)
    if min_conf is None and max_conf is None and not present:
        present = set(class_counts_of(item))
    return present


def filtered_counts(
    item: dict,
    min_conf: float | None = None,
    max_conf: float | None = None,
) -> dict[str, int]:
    if min_conf is None and max_conf is None:
        return class_counts_of(item)
    tallies: dict[str, int] = {}
    for det in detections_of(item):
        conf = detection_confidence(det)
        if min_conf is not None and conf < min_conf:
            continue
        if max_conf is not None and conf > max_conf:
            continue
        label = detection_label(det)
        if label:
            tallies[label] = tallies.get(label, 0) + 1
    return tallies


def image_id_for(path: str | Path) -> str:
    raw = str(Path(path).name).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def filename_of(item: dict) -> str:
    if item.get("filename"):
        return str(item["filename"])
    path = item.get("image_path") or ""
    return Path(str(path)).name or "image"


def processed_at_of(item: dict) -> str:
    return str(item.get("processed_at") or item.get("processed_timestamp") or "")


def highest_confidence_of(item: dict) -> float:
    if item.get("highest_confidence") is not None:
        try:
            return float(item["highest_confidence"])
        except (TypeError, ValueError):
            pass
    confs = [detection_confidence(d) for d in detections_of(item)]
    return max(confs) if confs else 0.0


def enrich_record(item: dict, model_used: str | None = None, elapsed: float | None = None) -> dict:
    """Add new fields without dropping the original YOLO payload."""
    path = item.get("image_path") or ""
    detections = detections_of(item)
    counts = class_counts_of(item)
    objects = []
    for det in detections:
        objects.append(
            {
                "label": detection_label(det) or det.get("class"),
                "confidence": detection_confidence(det),
                "bbox": det.get("bbox"),
            }
        )
    item.setdefault("image_id", image_id_for(path))
    item.setdefault("filename", filename_of(item))
    item["objects"] = objects
    item["object_counts"] = counts
    item["highest_confidence"] = highest_confidence_of(item)
    if model_used:
        item["model_used"] = model_used
    else:
        item.setdefault("model_used", "yolo11")
    if elapsed is not None:
        item["processing_time"] = float(elapsed)
    item.setdefault("processed_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    item.setdefault("has_embedding", False)
    return item


def unique_classes_from(metadata: list[dict]) -> list[str]:
    names: set[str] = set()
    for item in metadata:
        names.update(class_counts_of(item))
        for det in detections_of(item):
            label = detection_label(det)
            if label:
                names.add(label)
    return sorted(names)
