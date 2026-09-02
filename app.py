"""
YOLO Vision Search — Streamlit UI
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.collections import load_registry, seed_from_existing_metadata, upsert_collection
from src.config import load_config
from src.embedding_service import EmbeddingService, load_embeddings, save_embeddings
from src.inference import YOLOv11Inference
from src.metadata_service import (
    class_counts_of,
    detections_of,
    enrich_record,
    filename_of,
    highest_confidence_of,
    image_id_for,
    processed_at_of,
)
from src.query_parser import ast_to_text, builder_to_query, parse_query
from src.search_engine import SearchFilters, run_search
from src.utils import (
    collect_directory_images,
    ensure_processed_dir,
    format_unresolved_image_diagnostics,
    get_unique_classes_counts,
    infer_source_dir,
    load_metadata,
    resolve_image_path,
    resolve_project_path,
    save_metadata,
)


CONFIG = load_config()
MODEL_PATH = CONFIG["model"]["yolo_model"]
DEFAULT_CONFIDENCE = float(CONFIG["model"]["conf_threshold"])
IMAGE_EXTENSIONS = tuple(CONFIG["data"]["image_extension"])
PROJECT_ROOT = Path(__file__).resolve().parent
NAV_PAGES = ["Overview", "Analyze Image", "Collections", "Image Search", "History", "About"]
PAGE_COPY = {
    "Overview": ("Overview", "Detection, collections, and visual search"),
    "Analyze Image": ("Analyze Image", "Run YOLOv11 on a single image"),
    "Collections": ("Collections", "Index folders with YOLO and optional CLIP embeddings"),
    "Image Search": ("Image Search", "Boolean object search and semantic similarity"),
    "History": ("History", "Session analyses and searches from this browser run"),
    "About": ("About", "Product, stack, and architecture"),
}
HISTORY_LIMIT = 20
SEARCH_HISTORY_LIMIT = 30
MAX_UPLOAD_MB = 20
BOX_COLORS = [
    "#3B82F6", "#22C55E", "#F59E0B", "#A78BFA",
    "#F43F5E", "#06B6D4", "#84CC16", "#FB7185",
]

st.set_page_config(
    page_title="YOLO Vision Search",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "IBM Plex Sans", sans-serif;
}

.stApp {
    background: #0b1018;
    color: #e2e8f0;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.6rem;
    max-width: 1280px;
}

section[data-testid="stSidebar"] {
    background: #0a0f18;
    border-right: 1px solid #1c2533;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.15rem;
}

h1, h2, h3 {
    font-family: "IBM Plex Sans", sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

.brand-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #60a5fa;
}

.brand-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 0.2rem;
}

.brand-sub {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 0.15rem;
}

.app-header h1 {
    font-size: 1.35rem !important;
    font-weight: 650 !important;
    margin: 0 !important;
}

.header-sub {
    color: #94a3b8;
    font-size: 0.84rem;
    margin-top: 0.2rem;
}

.header-pills {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.pill {
    border: 1px solid #243044;
    background: #121826;
    color: #cbd5e1;
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.75rem;
}

.pill-live {
    border-color: rgba(34, 197, 94, 0.35);
    color: #86efac;
}

.status-mini {
    background: #121826;
    border: 1px solid #243044;
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
}

.status-mini .label {
    color: #94a3b8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.status-mini .row {
    color: #e2e8f0;
    font-size: 0.82rem;
    margin-top: 0.35rem;
}

.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
}

.hero {
    padding: 0.4rem 0 0.9rem 0;
}

.hero h1 {
    font-size: 2rem !important;
    font-weight: 650 !important;
    line-height: 1.2 !important;
    margin-bottom: 0.45rem !important;
}

.hero p {
    color: #94a3b8;
    font-size: 0.98rem;
    max-width: 42rem;
    line-height: 1.55;
}

.cap-card, .empty-state, .surface, .obj-chip, .about-card, .collection-card {
    background: #121826;
    border: 1px solid #243044;
    border-radius: 12px;
}

.cap-card {
    padding: 0.95rem 1rem;
    min-height: 132px;
}

.cap-card h3, .about-card h3, .empty-state h3, .collection-card h3 {
    font-size: 0.95rem !important;
    margin: 0.35rem 0 0.35rem 0 !important;
}

.cap-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: rgba(59, 130, 246, 0.12);
    color: #60a5fa;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
}

.muted {
    color: #94a3b8;
    font-size: 0.86rem;
    line-height: 1.5;
}

.empty-state, .surface, .collection-card {
    padding: 1.1rem 1.15rem;
}

.obj-chip {
    display: inline-block;
    vertical-align: top;
    padding: 0.65rem 0.8rem;
    margin: 0 0.4rem 0.45rem 0;
    min-width: 132px;
}

.obj-chip .name {
    color: #f8fafc;
    font-weight: 600;
    font-size: 0.88rem;
}

.obj-chip .meta {
    color: #94a3b8;
    font-size: 0.75rem;
    margin-top: 0.18rem;
    line-height: 1.4;
}

.result-shell {
    background: #121826;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.55rem;
    margin-bottom: 0.7rem;
}

.reason {
    color: #cbd5e1;
    font-size: 0.78rem;
    line-height: 1.45;
}

.flow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    color: #cbd5e1;
    font-size: 0.88rem;
}

.flow span {
    background: #0b1018;
    border: 1px solid #243044;
    border-radius: 8px;
    padding: 0.35rem 0.6rem;
}

.history-meta {
    color: #94a3b8;
    font-size: 0.84rem;
    line-height: 1.55;
}

[data-testid="stMetric"] {
    background: #121826;
    border: 1px solid #243044;
    border-radius: 12px;
    padding: 0.7rem 0.85rem;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.45);
}

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

div[data-testid="stSidebarNav"] { display: none; }

@media (max-width: 768px) {
    .hero h1 { font-size: 1.55rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "page": "Overview",
        "uploaded_image": None,
        "uploaded_name": None,
        "uploaded_size": None,
        "annotated_image": None,
        "detections": [],
        "analysis_time": 0.0,
        "history": [],
        "search_history": [],
        "collection_history": [],
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "embeddings": {},
        "active_collection": None,
        "search_results": [],
        "search_ran": False,
        "last_query": "",
        "last_mode": "Auto",
        "last_resolved_mode": "",
        "search_classes": [],
        "exclude_classes": [],
        "builder_combine": "AND",
        "last_search_classes": [],
        "uploader_key": 0,
        "last_model_name": MODEL_PATH,
        "last_device": None,
        "show_boxes": True,
        "show_labels": True,
        "show_confidence": True,
        "detail_hit": None,
        "confirm_clear_history": False,
        "fullscreen": None,
        "filter_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.page in {"Home", "Search"}:
        st.session_state.page = "Image Search" if st.session_state.page == "Search" else "Overview"
    seed_from_existing_metadata()


init_session_state()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_short() -> str:
    return "GPU" if torch.cuda.is_available() else "CPU"


def device_label() -> str:
    if torch.cuda.is_available():
        try:
            return f"GPU · {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU"
    return "CPU"


def format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "—"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} MB"


def show_issue(title: str, what: str, how: str) -> None:
    st.error(title)
    st.markdown(f"{what}\n\n**How to fix:** {how}")


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <h3>{title}</h3>
            <p class="muted">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_detections(detections: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in detections:
        name = str(item.get("class") or item.get("label") or "object")
        grouped[name].append(float(item.get("confidence") or 0.0))
    return {
        name: {
            "count": len(scores),
            "avg": sum(scores) / len(scores),
            "max": max(scores),
        }
        for name, scores in grouped.items()
    }


def render_detection_summary(detections: list[dict], expandable: bool = False) -> None:
    summary = summarize_detections(detections)
    if not summary:
        render_empty_state(
            "No objects detected",
            "Nothing met the confidence threshold. Lower the threshold and analyze again.",
        )
        return
    chips = []
    for name, info in summary.items():
        chips.append(
            f"""<div class="obj-chip">
                <div class="name">{name.title()}</div>
                <div class="meta">{info["count"]} detected<br>Highest confidence {info["max"]:.0%}</div>
            </div>"""
        )
    st.markdown("".join(chips), unsafe_allow_html=True)
    if expandable:
        with st.expander("Detection details"):
            st.json(
                [
                    {
                        "class": d.get("class") or d.get("label"),
                        "confidence": d.get("confidence"),
                        "bbox": d.get("bbox"),
                    }
                    for d in detections
                ]
            )


def make_thumbnail(image: Image.Image, max_width: int = 220) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((max_width, max_width))
    return thumb


def _font(size: int = 14):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    show_boxes: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    highlight: set[str] | None = None,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    if not detections or (not show_boxes and not show_labels and not show_conf):
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    color_map: dict[str, str] = {}
    for det in detections:
        cls = str(det.get("class") or det.get("label") or "object")
        if highlight is not None and cls.lower() not in {h.lower() for h in highlight}:
            continue
        if cls not in color_map:
            color_map[cls] = BOX_COLORS[len(color_map) % len(BOX_COLORS)]
        color = color_map[cls]
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if show_labels or show_conf:
            parts = []
            if show_labels:
                parts.append(cls)
            if show_conf:
                parts.append(f"{float(det['confidence']):.0%}")
            label = " ".join(parts)
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            pad = 4
            draw.rectangle(
                [text_bbox[0], text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + 2],
                fill=color,
            )
            draw.text((x1 + 2, y1 - pad + 1), label, fill="white", font=font)
    return canvas


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    device = get_device()
    model = YOLO(model_path)
    return model, device


@st.cache_resource(show_spinner="Loading CLIP semantic model…")
def load_embedding_service(device: str) -> EmbeddingService:
    return EmbeddingService(device=device)


def parse_yolo_result(result) -> tuple[Image.Image, list[dict]]:
    plotted = result.plot()
    annotated = Image.fromarray(plotted[:, :, ::-1])
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class": result.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
            )
    return annotated, detections


def analyze_pil_image(image: Image.Image, model_path: str, confidence: float):
    model, device = load_yolo_model(model_path)
    started = time.time()
    results = model.predict(
        source=image,
        conf=confidence,
        device=device,
        verbose=False,
    )
    elapsed = time.time() - started
    annotated, detections = parse_yolo_result(results[0])
    return annotated, detections, elapsed, device


def reset_analysis() -> None:
    st.session_state.uploaded_image = None
    st.session_state.uploaded_name = None
    st.session_state.uploaded_size = None
    st.session_state.annotated_image = None
    st.session_state.detections = []
    st.session_state.analysis_time = 0.0
    st.session_state.fullscreen = None
    st.session_state.uploader_key += 1


def apply_metadata(metadata: list, collection: dict | None = None, embeddings: dict | None = None) -> None:
    for item in metadata:
        enrich_record(item)
    st.session_state.metadata = metadata
    unique_classes, count_options = get_unique_classes_counts(metadata)
    st.session_state.unique_classes = unique_classes
    st.session_state.count_options = count_options
    st.session_state.search_results = []
    st.session_state.search_ran = False
    st.session_state.detail_hit = None
    st.session_state.active_collection = collection
    if embeddings is not None:
        st.session_state.embeddings = embeddings
    elif collection and collection.get("embeddings_path"):
        st.session_state.embeddings = load_embeddings(collection["embeddings_path"])
    else:
        st.session_state.embeddings = {}


def autoload_default_collection() -> None:
    if st.session_state.metadata:
        return
    registry = load_registry()
    if not registry:
        return
    preferred = [row for row in registry if row.get("search_available")] or list(registry)
    for item in preferred:
        path = item.get("metadata_path")
        if not path:
            continue
        try:
            metadata = load_metadata(path)
            embeddings = load_embeddings(item.get("embeddings_path") or "")
            apply_metadata(metadata, item, embeddings)
            return
        except Exception:
            continue


def collection_image_roots() -> list[Path]:
    col = st.session_state.get("active_collection") or {}
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | str | None) -> None:
        if not path:
            return
        resolved = resolve_project_path(path)
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    _add(col.get("source_dir"))
    _add(
        infer_source_dir(
            metadata=st.session_state.get("metadata") or [],
            metadata_path=col.get("metadata_path"),
            explicit=col.get("source_dir"),
        )
    )
    meta_path = col.get("metadata_path")
    if meta_path:
        meta = resolve_project_path(meta_path)
        _add(meta.parent if meta.is_file() or meta.suffix else meta)
    return roots


def set_nav(page: str) -> None:
    st.session_state.page = page
    st.session_state.fullscreen = None


def restore_history_item(item: dict) -> None:
    st.session_state.uploaded_image = item.get("image")
    st.session_state.uploaded_name = item.get("filename")
    st.session_state.detections = item.get("detections") or []
    st.session_state.analysis_time = item.get("processing_time", 0.0)
    st.session_state.annotated_image = item.get("annotated")
    st.session_state.last_model_name = item.get("model_name", MODEL_PATH)
    st.session_state.last_device = item.get("device")
    st.session_state.page = "Analyze Image"


def remove_history_item(item_id: str) -> None:
    st.session_state.history = [
        item for item in st.session_state.history if item.get("id") != item_id
    ]


def reuse_search(entry: dict) -> None:
    st.session_state.page = "Image Search"
    st.session_state.query_text = entry.get("query", "")
    st.session_state.search_mode_ui = entry.get("mode", "Auto")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-kicker">YOLO Vision</div>
            <div class="brand-title">Vision Search</div>
            <div class="brand-sub">Visual intelligence workspace</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        for page in NAV_PAGES:
            st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if st.session_state.page == page else "secondary",
                on_click=set_nav,
                args=(page,),
            )
        st.write("")
        st.markdown(
            f"""
            <div class="status-mini">
                <div class="label">AI system status</div>
                <div class="row"><span class="dot"></span>Ready</div>
                <div class="row">Model: {Path(MODEL_PATH).name}</div>
                <div class="row">Device: {device_label()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header() -> None:
    title, subtitle = PAGE_COPY.get(st.session_state.page, (st.session_state.page, ""))
    left, right = st.columns([3.2, 2])
    with left:
        st.markdown(
            f"""
            <div class="app-header" style="border:none;margin:0;padding:0;">
                <div>
                    <h1>{title}</h1>
                    <div class="header-sub">{subtitle}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="header-pills">
                <span class="pill pill-live">Model ready</span>
                <span class="pill">{device_short()}</span>
                <span class="pill">{Path(MODEL_PATH).stem}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Help", expanded=False):
            st.caption(
                "YOLO matches exact object classes. CLIP similarity is used only for "
                "descriptive queries. Boolean AND / OR / NOT applies to object labels, not clothing."
            )
    st.markdown(
        "<div style='border-bottom:1px solid #1c2533;margin:0.4rem 0 1rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>AI-Powered Visual Search</h1>
            <p>Analyze images, index visual collections, and search your data using objects, metadata, and natural language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1.05, 1.05, 2.4])
    with c1:
        st.button(
            "Analyze an Image",
            type="primary",
            use_container_width=True,
            on_click=set_nav,
            args=("Analyze Image",),
        )
    with c2:
        st.button(
            "Search Collection",
            use_container_width=True,
            on_click=set_nav,
            args=("Image Search",),
        )

    st.write("")
    st.markdown("**Capabilities**")
    cards = [
        ("01", "Object Detection", "YOLO bounding boxes, labels, and confidence scores."),
        ("02", "Visual Search", "Exact AND / OR / NOT matching on stored object metadata."),
        ("03", "Semantic Search", "CLIP similarity for descriptions such as clothing or color."),
        ("04", "Collection Indexing", "Process folders once; search metadata and embeddings later."),
    ]
    cols = st.columns(4)
    for col, (idx, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="cap-card">
                    <div class="cap-icon">{idx}</div>
                    <h3>{title}</h3>
                    <p class="muted">{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_analyze_page() -> None:
    left, right = st.columns([1.35, 1])
    with left:
        uploaded = st.file_uploader(
            "Drop an image or browse",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_key}",
            help=f"Accepted formats: JPG, JPEG, PNG. Maximum size {MAX_UPLOAD_MB} MB.",
        )
        st.caption(f"Accepted formats: JPG, JPEG, PNG. Maximum file size: {MAX_UPLOAD_MB} MB.")
        preview = None
        if uploaded is not None:
            size = getattr(uploaded, "size", None)
            if size and size > MAX_UPLOAD_MB * 1024 * 1024:
                show_issue(
                    "File is too large",
                    f"This file is {format_bytes(size)}.",
                    f"Use an image under {MAX_UPLOAD_MB} MB.",
                )
                return
            try:
                preview = Image.open(uploaded).convert("RGB")
                st.caption(
                    f"Selected: {uploaded.name} · {preview.width} × {preview.height} · "
                    f"{format_bytes(getattr(uploaded, 'size', None))}"
                )
            except Exception:
                show_issue(
                    "Unsupported or unreadable file",
                    "The file could not be opened as a still image.",
                    "Export it as JPG or PNG and upload again.",
                )
                return

    with right:
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            help="Detections below this score are discarded before results are shown.",
        )
        st.caption(f"Model `{MODEL_PATH}` · {device_label()}")
        analyze_clicked = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None and st.session_state.uploaded_image is None,
        )

    if analyze_clicked and uploaded is None and st.session_state.uploaded_image is None:
        show_issue(
            "No image selected",
            "Analysis needs an uploaded JPG, JPEG, or PNG file.",
            "Drop a file into the upload area, then run analysis.",
        )
        return

    if analyze_clicked and uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        try:
            with st.status("AI is analyzing your image", expanded=True) as status:
                st.write("Image processing")
                st.write("Detecting objects")
                annotated, detections, elapsed, device = analyze_pil_image(
                    image, MODEL_PATH, confidence
                )
                st.write("Extracting results")
                st.session_state.uploaded_image = image
                st.session_state.uploaded_name = uploaded.name
                st.session_state.uploaded_size = getattr(uploaded, "size", None)
                st.session_state.annotated_image = annotated
                st.session_state.detections = detections
                st.session_state.analysis_time = elapsed
                st.session_state.last_model_name = MODEL_PATH
                st.session_state.last_device = device
                highest = max((d["confidence"] for d in detections), default=0.0)
                st.session_state.history.append(
                    {
                        "id": str(uuid.uuid4()),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "objects": len(detections),
                        "top_confidence": highest,
                        "processing_time": elapsed,
                        "thumbnail": make_thumbnail(image),
                        "image": image,
                        "detections": detections,
                        "annotated": annotated,
                        "model_name": MODEL_PATH,
                        "device": device,
                    }
                )
                st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
                status.update(label="Analysis complete", state="complete")
            st.success("Analysis complete. Metrics below are from this YOLO run.")
        except Exception as exc:
            message = str(exc).lower()
            if "cuda" in message or "gpu" in message:
                show_issue(
                    "YOLO inference failed on GPU",
                    str(exc),
                    "Confirm CUDA is available, or continue on CPU.",
                )
            elif any(token in message for token in ("weight", "model", "download", "no such file")):
                show_issue(
                    "Model loading failed",
                    str(exc),
                    f"Place `{MODEL_PATH}` in the project root or allow Ultralytics to download it.",
                )
            else:
                show_issue(
                    "YOLO inference failed",
                    str(exc),
                    "Check the image and model weights, then try again.",
                )
            return

    original = preview or st.session_state.uploaded_image
    show_results = (
        st.session_state.uploaded_image is not None
        and (
            uploaded is None
            or st.session_state.uploaded_name == getattr(uploaded, "name", None)
        )
    )

    if original is None:
        st.write("")
        render_empty_state(
            "No image analyzed yet.",
            "Upload an image to begin AI-powered object detection.",
        )
        return

    if show_results:
        detections = st.session_state.detections
        highest = max((d["confidence"] for d in detections), default=0.0)
        device = st.session_state.last_device or get_device()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Objects Detected", len(detections))
        m2.metric("Highest Confidence", f"{highest:.0%}" if detections else "—")
        m3.metric("Processing Time", f"{st.session_state.analysis_time:.2f}s")
        m4.metric("Model Used", f"{Path(st.session_state.last_model_name).stem} · {str(device).upper()}")

        t1, t2, t3, t4 = st.columns(4)
        toggle = st.toggle if hasattr(st, "toggle") else st.checkbox
        with t1:
            show_boxes = toggle("Bounding boxes", key="show_boxes")
        with t2:
            show_labels = toggle("Labels", key="show_labels")
        with t3:
            show_conf = toggle("Confidence", key="show_confidence")
        with t4:
            if st.button("Fullscreen result", use_container_width=True):
                st.session_state.fullscreen = "result"

        detection_view = draw_detections(
            st.session_state.uploaded_image,
            detections,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf,
        )
        st.session_state.annotated_image = detection_view
    else:
        detection_view = None

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Original Image")
        st.image(original, use_container_width=True)
        if st.button("Fullscreen original"):
            st.session_state.fullscreen = "original"
    with col_b:
        st.caption("Detection Result")
        if detection_view is not None:
            st.image(detection_view, use_container_width=True)
        else:
            render_empty_state(
                "Waiting for inference",
                "Run Analyze Image to draw bounding boxes from YOLO.",
            )

    if st.session_state.fullscreen:
        target = original if st.session_state.fullscreen == "original" else st.session_state.annotated_image
        if target is not None:
            st.image(target, caption="Expanded view", use_container_width=True)
            if st.button("Close expanded view"):
                st.session_state.fullscreen = None
                st.rerun()

    if show_results:
        st.write("")
        st.markdown("**Detected Objects**")
        render_detection_summary(st.session_state.detections, expandable=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            buffer = io.BytesIO()
            (st.session_state.annotated_image or original).save(buffer, format="PNG")
            st.download_button(
                "Download result",
                data=buffer.getvalue(),
                file_name="yolo_detection_result.png",
                mime="image/png",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download JSON",
                data=json.dumps(st.session_state.detections, indent=2),
                file_name="detections.json",
                mime="application/json",
                use_container_width=True,
            )
        with d3:
            if st.button("Analyze another image", use_container_width=True):
                reset_analysis()
                st.rerun()


def process_collection(
    name: str,
    directory: Path,
    generate_embeddings: bool,
    skip_existing: bool,
) -> None:
    image_directory = Path(directory)
    inferencer = YOLOv11Inference(MODEL_PATH, device=get_device())
    paths = collect_directory_images(image_directory, inferencer.extensions)
    if not paths:
        show_issue(
            "No images found",
            f"No {', '.join(IMAGE_EXTENSIONS)} files in that folder.",
            "Use a directory with images in the top level, not nested albums.",
        )
        return

    existing_by_name = {}
    prior_path = ensure_processed_dir(image_directory) / "metadata.json"
    prior_rows = []
    if skip_existing and prior_path.exists():
        try:
            prior_rows = load_metadata(str(prior_path))
        except Exception:
            prior_rows = []
    elif skip_existing and st.session_state.metadata:
        prior_rows = st.session_state.metadata
    for item in prior_rows:
        existing_by_name[filename_of(item)] = item

    embedder = None
    if generate_embeddings:
        embedder = load_embedding_service(get_device())
        if not embedder.available:
            st.warning(embedder.error or "Semantic model unavailable. Indexing YOLO metadata only.")
            embedder = None

    processed_dir = ensure_processed_dir(image_directory)
    embeddings_path = processed_dir / "embeddings.npz"
    embedding_map = load_embeddings(embeddings_path) if embeddings_path.exists() else {}
    if st.session_state.embeddings:
        embedding_map.update(st.session_state.embeddings)

    metadata = []
    progress = st.progress(0.0)
    label = st.empty()
    yolo_ok = 0
    embed_ok = 0

    def _ensure_embedding(record: dict, img_path: Path) -> None:
        nonlocal embed_ok
        if not embedder:
            return
        iid = record.get("image_id") or image_id_for(img_path)
        record["image_id"] = iid
        if iid in embedding_map or filename_of(record) in embedding_map:
            record["has_embedding"] = True
            embed_ok += 1
            return
        try:
            vec = embedder.encode_image(img_path)
            record["has_embedding"] = True
            embedding_map[iid] = vec
            embed_ok += 1
        except Exception:
            record["has_embedding"] = False

    with st.status("Processing collection", expanded=True) as status:
        st.write("YOLO detection — running")
        if embedder:
            st.write("Embedding generation — running")
        else:
            st.write("Embedding generation — skipped")
        st.write("Metadata indexing — pending")

        for i, img_path in enumerate(paths, start=1):
            label.caption(f"Processing image {i} of {len(paths)}")
            progress.progress(i / len(paths))
            name_only = img_path.name
            if skip_existing and name_only in existing_by_name:
                record = existing_by_name[name_only]
                enrich_record(record)
                metadata.append(record)
                yolo_ok += 1
                _ensure_embedding(record, img_path)
                continue
            try:
                record = inferencer.process_image(img_path)
            except Exception:
                continue
            yolo_ok += 1
            _ensure_embedding(record, img_path)
            metadata.append(record)

        output_path = save_metadata(metadata, str(image_directory))
        embeddings_path = output_path.parent / "embeddings.npz"
        if embedding_map:
            import numpy as np

            ids = list(embedding_map.keys())
            stacked = np.vstack([embedding_map[i] for i in ids])
            save_embeddings(embeddings_path, ids, stacked)

        collection = upsert_collection(
            {
                "name": name or image_directory.name,
                "source_dir": str(image_directory),
                "metadata_path": str(output_path),
                "embeddings_path": str(embeddings_path),
                "image_count": len(paths),
                "indexed_count": len(metadata),
                "has_embeddings": embeddings_path.exists(),
                "semantic_available": embeddings_path.exists(),
                "search_available": True,
                "status": "ready",
                "model_used": MODEL_PATH,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        apply_metadata(metadata, collection, embedding_map)
        st.session_state.collection_history.append(
            {
                "id": str(uuid.uuid4()),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": collection.get("name"),
                "indexed": len(metadata),
                "embeddings": embed_ok,
            }
        )
        status.update(label="Collection ready", state="complete")
        st.write(f"YOLO detection — complete ({yolo_ok} images)")
        st.write(
            "Embedding generation — complete"
            if embedder
            else "Embedding generation — skipped"
        )
        st.write("Metadata indexing — complete")

    progress.empty()
    label.empty()
    st.success(f"Indexed {len(metadata)} images in “{collection.get('name')}”.")


def render_collections_page() -> None:
    st.markdown("**Process a collection**")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        col_name = st.text_input("Collection name", placeholder="Street Images")
        image_dir = st.text_input(
            "Image directory",
            placeholder="Folder of JPG / PNG files",
            help="Top-level folder only. Nested albums are not scanned.",
        )
        generate_emb = st.checkbox(
            "Generate CLIP embeddings (required for descriptive semantic search)",
            value=True,
        )
        skip_existing = st.checkbox("Skip images already indexed in the loaded collection", value=True)
        if st.button("Process collection", type="primary", disabled=not str(image_dir).strip()):
            directory = resolve_project_path(image_dir)
            if not directory.exists() or not directory.is_dir():
                show_issue(
                    "Invalid image directory",
                    "The path is missing or is not a folder.",
                    "Enter a folder that contains JPG, JPEG, or PNG files.",
                )
            else:
                try:
                    process_collection(
                        col_name.strip() or directory.name,
                        directory,
                        generate_emb,
                        skip_existing,
                    )
                except Exception as exc:
                    show_issue(
                        "Directory processing failed",
                        str(exc),
                        "Confirm the folder and model weights are accessible, then retry.",
                    )

    with c2:
        metadata_path = st.text_input("Or load metadata.json", placeholder="path/to/metadata.json")
        if st.button("Load metadata", disabled=not str(metadata_path).strip()):
            path = Path(metadata_path)
            if path.exists() and path.is_dir():
                show_issue(
                    "Missing metadata",
                    "A folder was provided instead of a JSON file.",
                    "Point this field at metadata.json.",
                )
            else:
                try:
                    metadata = load_metadata(metadata_path)
                    meta_file = resolve_project_path(metadata_path)
                    emb_path = meta_file.parent / "embeddings.npz"
                    embeddings = load_embeddings(emb_path) if emb_path.exists() else {}
                    image_directory = infer_source_dir(
                        metadata=metadata,
                        metadata_path=meta_file,
                    )
                    collection = upsert_collection(
                        {
                            "name": meta_file.parent.name,
                            "source_dir": str(image_directory),
                            "metadata_path": str(meta_file),
                            "embeddings_path": str(emb_path),
                            "image_count": len(metadata),
                            "indexed_count": len(metadata),
                            "has_embeddings": bool(embeddings),
                            "semantic_available": bool(embeddings),
                            "search_available": True,
                            "status": "ready",
                        }
                    )
                    apply_metadata(metadata, collection, embeddings)
                    st.success(f"Loaded {len(metadata)} indexed images.")
                except FileNotFoundError:
                    show_issue(
                        "Missing metadata",
                        "No metadata file was found at that location.",
                        "Process a collection first, or select an existing metadata.json.",
                    )
                except json.JSONDecodeError:
                    show_issue(
                        "Metadata could not be read",
                        "The file is not valid JSON.",
                        "Regenerate metadata by processing the image directory.",
                    )
                except Exception as exc:
                    show_issue(
                        "Metadata could not be read",
                        str(exc),
                        "Use a metadata file generated by this application.",
                    )

    st.write("")
    st.markdown("**Saved collections**")
    registry = load_registry()
    if not registry:
        render_empty_state(
            "No collections yet.",
            "Process a folder or load metadata.json to make a collection searchable.",
        )
        return

    for item in registry:
        semantic = "Yes" if item.get("semantic_available") or item.get("has_embeddings") else "No"
        st.markdown(
            f"""
            <div class="collection-card">
                <h3>{item.get("name", "Untitled")}</h3>
                <p class="muted">
                    {item.get("indexed_count", 0)} indexed ·
                    Status: {item.get("status", "unknown")} ·
                    Semantic search: {semantic}<br>
                    Created {item.get("created_at", "—")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Open in Search", key=f"open_{item.get('id')}", use_container_width=True):
                try:
                    metadata = load_metadata(item["metadata_path"])
                    embeddings = load_embeddings(item.get("embeddings_path") or "")
                    apply_metadata(metadata, item, embeddings)
                    st.session_state.page = "Image Search"
                    st.rerun()
                except Exception as exc:
                    show_issue("Could not open collection", str(exc), "Re-process the folder or check the metadata path.")
        with b2:
            st.caption(item.get("metadata_path", ""))


def _current_filters(known_classes: list[str]) -> SearchFilters:
    with st.expander("Filters", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
            min_objects = st.number_input("Minimum objects", min_value=0, value=0)
            result_limit = st.slider("Result limit", 6, 120, 48, 6)
        with f2:
            max_conf = st.slider("Maximum confidence", 0.0, 1.0, 1.0, 0.05)
            max_objects = st.number_input("Maximum objects", min_value=0, value=0)
            sim = st.slider("Semantic similarity threshold", 0.05, 0.8, 0.22, 0.01)
        class_filter = st.multiselect("Restrict to object classes", known_classes)
        bounds = {}
        selected = st.session_state.get("search_classes") or []
        if selected:
            st.caption("Object count bounds apply to builder include-classes.")
            for cls in selected:
                observed = st.session_state.count_options.get(cls) or [1]
                max_seen = max(int(v) for v in observed)
                a, b = st.columns(2)
                with a:
                    lo = st.number_input(f"{cls} min", 1, max_seen, 1, key=f"cmin_{cls}_{st.session_state.filter_nonce}")
                with b:
                    hi = st.number_input(
                        f"{cls} max",
                        int(lo),
                        max(max_seen, int(lo)),
                        max_seen,
                        key=f"cmax_{cls}_{st.session_state.filter_nonce}",
                    )
                bounds[cls] = (int(lo), int(hi))
    return SearchFilters(
        min_confidence=min_conf if min_conf > 0 else None,
        max_confidence=max_conf if max_conf < 1 else None,
        min_objects=int(min_objects) if min_objects else None,
        max_objects=int(max_objects) if max_objects else None,
        class_filter=class_filter,
        similarity_threshold=float(sim),
        result_limit=int(result_limit),
        count_bounds=bounds,
    )


def execute_search(query: str, mode: str, filters: SearchFilters) -> None:
    if not st.session_state.metadata:
        show_issue(
            "No indexed collection found",
            "Search needs YOLO metadata from a processed folder.",
            "Open Collections, process a directory, or load metadata.json.",
        )
        return
    query = (query or "").strip()
    if not query:
        show_issue("Empty query", "Type a query or use the boolean builder.", "Example: person AND motorcycle")
        return

    parsed_preview = parse_query(query, st.session_state.unique_classes)
    needs_semantic = mode.lower() in {"semantic", "hybrid"} or (
        mode.lower() == "auto" and parsed_preview.mode_hint in {"semantic", "hybrid"}
    )

    embedder = None
    query_vec = None
    semantic_available = False
    semantic_error = None
    if needs_semantic:
        embedder = load_embedding_service(get_device())
        semantic_available = bool(embedder.available)
        semantic_error = embedder.error
        if semantic_available:
            try:
                query_vec = embedder.encode_text(query)
            except Exception as exc:
                semantic_available = False
                semantic_error = str(exc)

    hits, parsed, resolved, warning = run_search(
        query=query,
        metadata=st.session_state.metadata,
        known_classes=st.session_state.unique_classes,
        mode=mode,
        embeddings=st.session_state.embeddings or {},
        query_vec=query_vec,
        filters=filters,
        semantic_available=semantic_available,
        semantic_error=semantic_error,
    )
    if parsed.error and not hits:
        show_issue("Invalid boolean query", parsed.error, "Use known YOLO class names with AND, OR, NOT, and parentheses.")
        return
    if warning and not hits:
        show_issue(
            "Search could not be completed",
            warning,
            "For object names use Structured or Auto. For clothing and color, index CLIP embeddings.",
        )
        st.session_state.search_results = []
        st.session_state.search_ran = True
        st.session_state.last_query = query
        st.session_state.last_mode = mode
        st.session_state.last_resolved_mode = resolved
        return
    if warning:
        st.warning(warning)

    st.session_state.search_results = [
        {
            "item": h.item,
            "score": h.score,
            "match_type": h.match_type,
            "reasons": h.reasons,
            "similarity": h.similarity,
        }
        for h in hits
    ]
    st.session_state.search_ran = True
    st.session_state.last_query = query
    st.session_state.last_mode = mode
    st.session_state.last_resolved_mode = resolved
    st.session_state.last_search_classes = parsed.extracted_classes
    st.session_state.detail_hit = None
    st.session_state.search_history.append(
        {
            "id": str(uuid.uuid4()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "mode": mode,
            "resolved": resolved,
            "results": len(hits),
        }
    )
    st.session_state.search_history = st.session_state.search_history[-SEARCH_HISTORY_LIMIT:]


def render_image_card(hit: dict, index: int, highlight: set[str]) -> None:
    item = hit["item"]
    filename = filename_of(item)
    class_counts = class_counts_of(item)
    total = item.get("total_objects", sum(class_counts.values()) if class_counts else 0)
    resolved = resolve_image_path(item.get("image_path", ""), extra_roots=collection_image_roots())
    collection_name = (st.session_state.active_collection or {}).get("name", "Loaded collection")

    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    if resolved is None:
        st.caption(f"{filename} — file not found on this machine")
        st.code(
            format_unresolved_image_diagnostics(
                item.get("image_path", ""),
                extra_roots=collection_image_roots(),
                filename=filename,
            ),
            language=None,
        )
    else:
        try:
            thumb = Image.open(resolved)
            thumb.thumbnail((640, 480))
            st.image(thumb, use_container_width=True)
        except Exception:
            st.caption(f"Could not open {filename}")
    tags = ", ".join(list(class_counts.keys())[:4]) or "No objects"
    extra = max(0, len(class_counts) - 4)
    if extra:
        tags += f" +{extra}"
    st.markdown(f"**{filename}**")
    st.caption(f"{tags}")
    score_txt = f"{hit['similarity']:.0%} similarity" if hit.get("similarity") is not None else hit.get("match_type")
    st.caption(f"{collection_name} · {total} objects · {score_txt}")
    reasons = hit.get("reasons") or []
    if reasons:
        st.markdown(
            "<div class='reason'>" + "<br>".join(reasons[:3]) + "</div>",
            unsafe_allow_html=True,
        )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("View details", key=f"view_{index}_{filename}", use_container_width=True):
            st.session_state.detail_hit = hit
            st.rerun()
    with b2:
        if resolved and resolved.exists():
            st.download_button(
                "Download",
                data=resolved.read_bytes(),
                file_name=filename,
                mime="image/jpeg",
                key=f"dl_{index}_{filename}",
                use_container_width=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_detail(hit: dict) -> None:
    item = hit["item"]
    filename = filename_of(item)
    resolved = resolve_image_path(item.get("image_path", ""), extra_roots=collection_image_roots())
    detections = detections_of(item)
    highlight = set(st.session_state.last_search_classes or [])

    top = st.columns([6, 1])
    with top[0]:
        st.markdown(f"**{filename}**")
        st.caption(item.get("image_path", filename))
    with top[1]:
        if st.button("Close"):
            st.session_state.detail_hit = None
            st.rerun()

    st.markdown("**Match details**")
    st.caption(f"Query: {st.session_state.last_query}")
    st.caption(
        f"Mode: {st.session_state.last_resolved_mode or st.session_state.last_mode} · "
        f"{hit.get('match_type')}"
    )
    for reason in hit.get("reasons") or []:
        st.write(reason)
    if hit.get("similarity") is not None:
        st.info(
            f"Semantic similarity {hit['similarity']:.0%}. "
            "This is CLIP visual-language similarity, not a guarantee of clothing or color attributes."
        )

    if resolved is None:
        show_issue(
            "Image file not found",
            format_unresolved_image_diagnostics(
                item.get("image_path", ""),
                extra_roots=collection_image_roots(),
                filename=filename,
            ),
            "Keep original images under data/raw (or the collection source_dir), then reload the collection.",
        )
        return

    original = Image.open(resolved).convert("RGB")
    detected = draw_detections(original, detections, highlight=highlight or None)
    left, right = st.columns(2)
    with left:
        st.caption("Original image")
        st.image(original, use_container_width=True)
    with right:
        st.caption("Detection image")
        st.image(detected, use_container_width=True)

    st.markdown("**Detected objects**")
    render_detection_summary(detections, expandable=True)
    counts = class_counts_of(item)
    if counts:
        st.caption("Object counts: " + ", ".join(f"{k} × {v}" for k, v in counts.items()))
    meta_bits = [
        f"ID {item.get('image_id', '—')}",
        f"Processed {processed_at_of(item) or '—'}",
        f"Highest confidence {highest_confidence_of(item):.0%}",
        f"Model {item.get('model_used', MODEL_PATH)}",
    ]
    st.caption(" · ".join(meta_bits))
def apply_boolean_query():
    include = st.session_state.get("builder_include", [])
    operator = st.session_state.get("builder_operator", "AND")
    exclude = st.session_state.get("builder_exclude", [])

    parts = []

    # Include objects
    if include:
        if len(include) == 1:
            parts.append(include[0])
        else:
            parts.append(f" {operator} ".join(include))

    # Exclude objects
    if exclude:
        exclude_query = " OR ".join(exclude)

        if parts:
            parts.append(f"NOT ({exclude_query})")
        else:
            parts.append(f"NOT ({exclude_query})")

    built = " AND ".join(parts)

    # Update BEFORE the widget is rendered
    st.session_state["query_text"] = built


def render_search_page() -> None:
    if not st.session_state.metadata:
        render_empty_state(
            "No indexed collection found.",
            "Process a collection first to enable image search.",
        )
        st.button("Go to Collections", on_click=set_nav, args=("Collections",))
        return

    col = st.session_state.active_collection or {}
    n_emb = len(st.session_state.embeddings or {})
    st.caption(
        f"{len(st.session_state.metadata)} images indexed"
        + (f" · {col.get('name')}" if col.get("name") else "")
        + (f" · {n_emb} CLIP embeddings" if n_emb else " · semantic search unavailable until embeddings are generated")
    )

    if "pending_query" in st.session_state:
        st.session_state["query_text"] = st.session_state.pop("pending_query")
    with st.form("search_form"):
        query = st.text_input(
            "Search",
            placeholder="Search images, objects, or descriptions...",
            key="query_text",
            help="Examples: person AND motorcycle · person NOT car · person wearing a black jacket",
        )
        mode = st.radio(
            "Search mode",
            ["Auto", "Structured", "Semantic", "Hybrid"],
            horizontal=True,
            key="search_mode_ui",
            help="Auto chooses structured vs semantic. Structured is exact YOLO labels. Semantic is CLIP similarity. Hybrid can combine both.",
        )
        run = st.form_submit_button("Search Images", type="primary")
    preview_q = (st.session_state.get("query_text") or query or "").strip()
    if preview_q:
        preview = parse_query(preview_q, st.session_state.unique_classes)
        interpreted = ast_to_text(preview.ast) if preview.ast else preview_q
        bits = [f"Interpreted as `{interpreted}`", f"mode {preview.mode_hint}"]
        if preview.extracted_classes:
            bits.append("objects: " + ", ".join(preview.extracted_classes))
        if preview.mode_hint in {"semantic", "hybrid"}:
            bits.append("descriptive terms use CLIP similarity, not YOLO attributes")
        st.caption(" · ".join(bits))
        if preview.error and preview.has_operators:
            st.warning(preview.error)

    with st.expander("Boolean builder", expanded=False):
        st.caption("Builds a structured query from known YOLO classes. Clothing and color are not YOLO classes.")
        include = st.multiselect(
            "Include objects",
            st.session_state.unique_classes,
            key="search_classes",
        )
        combine = st.radio("Combine includes with", ["AND", "OR"], horizontal=True, key="builder_combine")
        exclude = st.multiselect(
            "Exclude objects (NOT)",
            st.session_state.unique_classes,
            key="exclude_classes",
        )
        built = builder_to_query(include, combine, exclude)
        if built:
            st.code(built)
            if st.button("Use builder query"):
                st.session_state["pending_query"] = built
                st.rerun()

    filters = _current_filters(st.session_state.unique_classes)
    if run:
        execute_search(st.session_state.get("query_text") or query, mode, filters)

    if st.session_state.detail_hit is not None:
        render_result_detail(st.session_state.detail_hit)
        st.divider()

    if not st.session_state.search_ran:
        render_empty_state(
            "Ready to search.",
            "Try person AND motorcycle, car OR bicycle, person NOT car, or a description if embeddings exist.",
        )
        return

    hits = st.session_state.search_results
    st.markdown(
        f"**Showing {len(hits)} results**"
        + (f" · `{st.session_state.last_query}`" if st.session_state.last_query else "")
        + (f" · {st.session_state.last_resolved_mode}" if st.session_state.last_resolved_mode else "")
    )
    if not hits:
        render_empty_state(
            "No search results found.",
            "Adjust the query, filters, or similarity threshold. Semantic matches need CLIP embeddings.",
        )
        return

    highlight = set(st.session_state.last_search_classes or [])
    columns = st.columns(3)
    for index, hit in enumerate(hits):
        with columns[index % 3]:
            render_image_card(hit, index, highlight)


def render_history_page() -> None:
    st.caption("History is stored only for the current browser session, not on disk.")
    tab_a, tab_s, tab_c = st.tabs(["Image analysis", "Searches", "Collections"])

    with tab_a:
        if not st.session_state.history:
            render_empty_state(
                "No image analyzed yet.",
                "Upload an image to begin AI-powered object detection. Session history will appear here.",
            )
            st.button("Go to Analyze Image", on_click=set_nav, args=("Analyze Image",))
        else:
            if st.session_state.confirm_clear_history:
                st.warning("Clear all session analysis history?")
                yes, no = st.columns(2)
                if yes.button("Confirm", type="primary"):
                    st.session_state.history = []
                    st.session_state.confirm_clear_history = False
                    st.rerun()
                if no.button("Cancel"):
                    st.session_state.confirm_clear_history = False
                    st.rerun()
            elif st.button("Clear analysis history"):
                st.session_state.confirm_clear_history = True
                st.rerun()

            for item in reversed(st.session_state.history):
                left, mid, right = st.columns([1, 2.4, 1.1])
                with left:
                    if item.get("thumbnail") is not None:
                        st.image(item["thumbnail"], use_container_width=True)
                with mid:
                    st.markdown(f"**{item.get('filename', 'Untitled')}**")
                    st.markdown(
                        f"""
                        <div class="history-meta">
                            {item.get("time", "")}<br>
                            Objects detected: {item.get("objects", 0)} ·
                            Highest confidence: {item.get("top_confidence", 0):.0%}<br>
                            Processing time: {item.get("processing_time", 0):.2f}s
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with right:
                    st.button(
                        "View analysis",
                        key=f"hist_view_{item.get('id')}",
                        use_container_width=True,
                        on_click=restore_history_item,
                        args=(item,),
                    )
                    st.button(
                        "Remove",
                        key=f"hist_del_{item.get('id')}",
                        use_container_width=True,
                        on_click=remove_history_item,
                        args=(item.get("id"),),
                    )
                st.divider()

    with tab_s:
        if not st.session_state.search_history:
            render_empty_state("No searches yet.", "Run a search to record it in this session.")
        else:
            if st.button("Clear search history"):
                st.session_state.search_history = []
                st.rerun()
            for entry in reversed(st.session_state.search_history):
                st.markdown(f"**{entry.get('query')}**")
                st.caption(
                    f"{entry.get('time')} · {entry.get('mode')} → {entry.get('resolved')} · "
                    f"{entry.get('results')} results"
                )
                if st.button("Reuse search", key=f"reuse_{entry.get('id')}"):
                    reuse_search(entry)
                    st.rerun()
                st.divider()

    with tab_c:
        if not st.session_state.collection_history:
            render_empty_state(
                "No collection processing in this session.",
                "Indexed collections still appear on the Collections page.",
            )
        else:
            for entry in reversed(st.session_state.collection_history):
                st.markdown(f"**{entry.get('name')}**")
                st.caption(
                    f"{entry.get('time')} · {entry.get('indexed')} images · "
                    f"{entry.get('embeddings')} embeddings"
                )


def render_about_page() -> None:
    st.markdown(
        """
        <div class="surface">
            <h3>Product overview</h3>
            <p class="muted">
                YOLO Vision Search is a visual intelligence workspace. YOLO detects objects.
                Structured metadata search answers exact boolean queries. CLIP embeddings
                rank descriptive similarity. Clothing and color are not YOLO attributes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Technology stack**")
    stack = ["YOLOv11", "Ultralytics", "OpenCLIP", "Python", "Streamlit", "PIL"]
    st.markdown(
        "<div class='flow'>" + "".join(f"<span>{name}</span>" for name in stack) + "</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Architecture**")
    st.markdown(
        """
        <div class="flow">
            <span>Image Input</span>
            <span>YOLO Inference</span>
            <span>CLIP Embedding</span>
            <span>Metadata Storage</span>
            <span>Query Parser</span>
            <span>Hybrid Search</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            """
            <div class="about-card" style="padding:1rem;">
                <h3>Detection</h3>
                <p class="muted">Live analysis and folder indexing use YOLO boxes, labels, and confidence scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="about-card" style="padding:1rem;">
                <h3>Structured search</h3>
                <p class="muted">AND, OR, NOT, and parentheses operate on detected object classes and counts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            """
            <div class="about-card" style="padding:1rem;">
                <h3>Semantic search</h3>
                <p class="muted">CLIP similarity is probabilistic. It does not certify a black jacket or red hat.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


autoload_default_collection()
render_sidebar()
render_header()

page = st.session_state.page
if page == "Overview":
    render_overview()
elif page == "Analyze Image":
    render_analyze_page()
elif page == "Collections":
    render_collections_page()
elif page == "Image Search":
    render_search_page()
elif page == "History":
    render_history_page()
elif page == "About":
    render_about_page()
else:
    render_overview()
