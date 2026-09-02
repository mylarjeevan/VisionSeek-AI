"""Structured, semantic, and hybrid search over YOLO metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.embedding_service import cosine_similarity
from src.metadata_service import (
    class_counts_of,
    classes_present,
    detections_of,
    filename_of,
    filtered_counts,
    highest_confidence_of,
    processed_at_of,
)
from src.query_parser import ParseResult, parse_query

CLIP_UNAVAILABLE = (
    "Descriptive search needs CLIP embeddings. YOLO metadata only stores object "
    "classes (person, backpack, car, …), not clothing color or garments. "
    "Process a collection with CLIP embeddings enabled, or run "
    "scripts/prepare_sample_data.py and scripts/build_clip_embeddings.py."
)


def _requires_clip(parsed: ParseResult) -> bool:
    return bool(parsed.unknown_terms) or parsed.mode_hint in {"semantic", "hybrid"}


@dataclass
class SearchFilters:
    min_confidence: float | None = None
    max_confidence: float | None = None
    min_objects: int | None = None
    max_objects: int | None = None
    class_filter: list[str] = field(default_factory=list)
    processed_after: str = ""
    processed_before: str = ""
    similarity_threshold: float = 0.22
    result_limit: int = 48
    count_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class SearchHit:
    item: dict
    score: float
    match_type: str
    reasons: list[str]
    similarity: float | None = None


def eval_ast(node: dict[str, Any], present: set[str], counts: dict[str, int], bounds: dict) -> bool:
    op = node.get("operator")
    if op == "TERM":
        label = str(node.get("value", "")).lower()
        count = counts.get(label, 0)
        if label not in present and count <= 0:
            return False
        if label in bounds:
            low, high = bounds[label]
            return low <= count <= high
        return count >= 1
    if op == "NOT":
        children = node.get("conditions") or []
        if not children:
            return True
        return not eval_ast(children[0], present, counts, bounds)
    if op == "AND":
        return all(eval_ast(child, present, counts, bounds) for child in node.get("conditions") or [])
    if op == "OR":
        return any(eval_ast(child, present, counts, bounds) for child in node.get("conditions") or [])
    return False


def explain_ast(node: dict[str, Any], present: set[str], counts: dict[str, int]) -> list[str]:
    reasons: list[str] = []

    def walk(n: dict[str, Any], negated: bool = False) -> None:
        op = n.get("operator")
        if op == "TERM":
            label = str(n.get("value", ""))
            count = counts.get(label, 0)
            if negated:
                if count == 0:
                    reasons.append(f"✓ NOT {label} (none detected)")
                else:
                    reasons.append(f"✗ Contains {label} ({count})")
            elif count:
                reasons.append(f"✓ {label.title()} detected ({count})")
            else:
                reasons.append(f"✗ {label.title()} not detected")
            return
        if op == "NOT":
            kids = n.get("conditions") or []
            if kids:
                walk(kids[0], negated=not negated)
            return
        for child in n.get("conditions") or []:
            walk(child, negated)

    walk(node)
    return reasons


def _passes_filters(item: dict, filters: SearchFilters, counts: dict[str, int]) -> bool:
    total = sum(counts.values()) or int(item.get("total_objects") or 0)
    if filters.min_objects is not None and total < filters.min_objects:
        return False
    if filters.max_objects is not None and total > filters.max_objects:
        return False
    if filters.class_filter:
        wanted = {c.lower() for c in filters.class_filter}
        if not wanted.intersection(set(counts) | classes_present(item)):
            return False
    stamp = processed_at_of(item)
    if filters.processed_after and stamp and stamp < filters.processed_after:
        return False
    if filters.processed_before and stamp and stamp > filters.processed_before:
        return False
    high = highest_confidence_of(item)
    if filters.min_confidence is not None and detections_of(item):
        if high < filters.min_confidence:
            # still allow if any remaining detection after per-box filter exists
            if not counts:
                return False
    return True


def structured_search(metadata: list[dict], ast: dict, filters: SearchFilters | None = None) -> list[SearchHit]:
    filters = filters or SearchFilters()
    hits: list[SearchHit] = []
    for item in metadata:
        counts = filtered_counts(item, filters.min_confidence, filters.max_confidence)
        present = set(counts)
        if not _passes_filters(item, filters, counts):
            continue
        if not eval_ast(ast, present, counts, filters.count_bounds):
            continue
        reasons = explain_ast(ast, present, counts)
        hits.append(
            SearchHit(
                item=item,
                score=1.0,
                match_type="exact object match",
                reasons=reasons or ["Matched structured object query"],
            )
        )
    return hits


def semantic_search(
    metadata: list[dict],
    embeddings: dict[str, np.ndarray],
    query_vec: np.ndarray,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    filters = filters or SearchFilters()
    ranked: list[SearchHit] = []
    for item in metadata:
        counts = filtered_counts(item, filters.min_confidence, filters.max_confidence)
        if not _passes_filters(item, filters, counts):
            continue
        image_id = str(item.get("image_id") or filename_of(item))
        vec = embeddings.get(image_id)

        if vec is None:
            vec = embeddings.get(filename_of(item))
        if vec is None:
            continue
        sim = cosine_similarity(query_vec, vec)
        if sim < filters.similarity_threshold:
            continue
        ranked.append(
            SearchHit(
                item=item,
                score=sim,
                match_type="semantic similarity",
                similarity=sim,
                reasons=[
                    f"Semantic similarity {sim:.0%} (CLIP visual-language match, not a YOLO clothing detector)",
                ],
            )
        )
    ranked.sort(key=lambda h: h.score, reverse=True)
    return ranked


def hybrid_search(
    metadata: list[dict],
    ast: dict | None,
    embeddings: dict[str, np.ndarray],
    query_vec: np.ndarray | None,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    filters = filters or SearchFilters()
    pool = metadata
    structured_hits = []
    if ast is not None:
        structured_hits = structured_search(metadata, ast, filters)
        pool = [h.item for h in structured_hits]
        if not pool:
            return []
    if query_vec is None or not embeddings:
        return structured_hits
    semantic_hits = semantic_search(pool, embeddings, query_vec, filters)
    by_id = {}
    for hit in semantic_hits:
        key = hit.item.get("image_id") or filename_of(hit.item)
        by_id[key] = hit
        if ast is not None:
            extra = explain_ast(
                ast,
                set(filtered_counts(hit.item, filters.min_confidence, filters.max_confidence)),
                filtered_counts(hit.item, filters.min_confidence, filters.max_confidence),
            )
            hit.match_type = "hybrid"
            hit.reasons = extra + hit.reasons
    if semantic_hits:
        return semantic_hits
    return structured_hits


def _finalize(
    hits: list[SearchHit],
    parsed: ParseResult,
    chosen: str,
    warning: str | None,
    filters: SearchFilters,
) -> tuple[list[SearchHit], ParseResult, str, str | None]:
    total = len(hits)
    limit = filters.result_limit
    if limit and total > limit:
        note = f"Showing {limit} of {total} matches."
        warning = f"{warning} {note}".strip() if warning else note
        hits = hits[:limit]
    return hits, parsed, chosen, warning


def run_search(
    query: str,
    metadata: list[dict],
    known_classes: list[str],
    mode: str,
    embeddings: dict[str, np.ndarray] | None = None,
    query_vec: np.ndarray | None = None,
    filters: SearchFilters | None = None,
    semantic_available: bool = False,
    semantic_error: str | None = None,
) -> tuple[list[SearchHit], ParseResult, str, str | None]:
    """Returns hits, parse result, resolved mode, warning."""
    filters = filters or SearchFilters()
    parsed = parse_query(query, known_classes)
    if parsed.error and parsed.mode_hint == "structured" and parsed.has_operators:
        return [], parsed, mode, parsed.error

    chosen = (mode or "auto").lower()
    warning = None
    if chosen == "auto":
        chosen = parsed.mode_hint or "structured"

    if chosen == "structured":
        if parsed.ast is None:
            return [], parsed, chosen, parsed.error or "Could not parse a structured object query."
        if parsed.error and parsed.unknown_terms:
            return [], parsed, chosen, parsed.error
        hits = structured_search(metadata, parsed.ast, filters)
        return _finalize(hits, parsed, chosen, warning, filters)

    if chosen == "semantic":
        if not semantic_available or query_vec is None:
            return [], parsed, chosen, semantic_error or "Semantic model unavailable."
        if not embeddings:
            return [], parsed, chosen, (
                "This collection has YOLO metadata but no CLIP embeddings. "
                "Re-process the collection to enable semantic search."
            )
        hits = semantic_search(metadata, embeddings, query_vec, filters)
        return _finalize(hits, parsed, chosen, warning, filters)

    # hybrid
    if parsed.error and parsed.has_operators and parsed.unknown_terms:
        return [], parsed, "hybrid", parsed.error
    if parsed.has_operators and parsed.ast and not parsed.unknown_terms:
        ast = parsed.ast
    elif parsed.extracted_classes:
        terms = [
            {"operator": "TERM", "value": c, "known": True}
            for c in parsed.extracted_classes
        ]
        ast = terms[0] if len(terms) == 1 else {"operator": "AND", "conditions": terms}
    else:
        ast = None
    if not semantic_available:
        if ast is None or _requires_clip(parsed):
            return [], parsed, "hybrid", semantic_error or CLIP_UNAVAILABLE
        warning = "Semantic model unavailable — used exact object match only."
        hits = structured_search(metadata, ast, filters)
        return _finalize(hits, parsed, "structured", warning, filters)
    if not embeddings:
        if ast is None or _requires_clip(parsed):
            return [], parsed, "hybrid", CLIP_UNAVAILABLE
        warning = "No embeddings stored — used exact object match only."
        hits = structured_search(metadata, ast, filters)
        return _finalize(hits, parsed, "structured", warning, filters)
    hits = hybrid_search(metadata, ast, embeddings, query_vec, filters)
    return _finalize(hits, parsed, "hybrid", warning, filters)
