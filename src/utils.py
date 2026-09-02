import json
import os
from pathlib import Path

from src.metadata_service import class_counts_of, detections_of, detection_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_COCO_DIR = PROJECT_ROOT / "data" / "raw" / "coco-val-2017-500"
DEFAULT_COCO_REL = "data/raw/coco-val-2017-500"


def _as_posix(path_str: str) -> str:
    return str(path_str).replace("\\", "/")


def portable_relpath(path: str | Path) -> str:
    """Store paths relative to the project when they live under data/ or PROJECT_ROOT."""
    raw = str(path or "").strip()
    if not raw:
        return raw
    posix = _as_posix(raw)
    lower = posix.lower()
    for marker in ("data/raw/", "data/processed/"):
        idx = lower.find(marker)
        if idx >= 0:
            return posix[idx:]
    try:
        resolved = Path(raw).expanduser().resolve()
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return posix


def configured_image_root() -> Path:
    """Project-relative or absolute directory that holds original images."""
    env = os.environ.get("YOLO_IMAGE_ROOT", "").strip()
    if env:
        raw = Path(env).expanduser()
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    try:
        from src.config import load_config

        configured = (load_config().get("data") or {}).get("image_root")
    except Exception:
        configured = None
    if configured:
        raw = Path(str(configured)).expanduser()
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    return DEFAULT_IMAGE_ROOT


def image_lookup_roots(extra_roots: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        try:
            key = str(path.expanduser().resolve()) if path.exists() else str(path)
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    for item in extra_roots or []:
        _add(Path(item))
    _add(configured_image_root())
    _add(DEFAULT_IMAGE_ROOT)
    _add(DEFAULT_COCO_DIR)
    _add(PROJECT_ROOT / "data" / "processed")
    return roots


def stored_relative_path(path_str: str | Path | None) -> str:
    """Project-relative tail such as data/raw/... extracted from any stored path."""
    raw = str(path_str or "").strip()
    if not raw:
        return ""
    posix = _as_posix(raw)
    lower = posix.lower()
    for marker in ("data/raw/", "data/processed/", "data/"):
        idx = lower.find(marker)
        if idx >= 0:
            return posix[idx:]
    path = Path(posix)
    if not path.is_absolute():
        return posix.lstrip("./")
    return ""


def infer_source_dir(
    metadata: list | None = None,
    metadata_path: str | Path | None = None,
    explicit: str | Path | None = None,
) -> Path:
    """Locate the image folder for a collection without machine-specific paths."""
    if explicit and str(explicit).strip():
        raw = str(explicit).strip()
        rel = stored_relative_path(raw)
        candidate = resolve_project_path(rel or raw)
        if candidate.is_file():
            candidate = candidate.parent
        return candidate

    parents: dict[str, int] = {}
    for item in metadata or []:
        if not isinstance(item, dict):
            continue
        stored = item.get("image_path") or ""
        rel = stored_relative_path(stored)
        if rel:
            parent = Path(rel).parent.as_posix()
            if parent and parent != ".":
                parents[parent] = parents.get(parent, 0) + 1
    if parents:
        best = max(parents, key=parents.get)
        candidate = resolve_project_path(best)
        if candidate.is_file():
            candidate = candidate.parent
        return candidate

    if metadata_path:
        meta = resolve_project_path(metadata_path)
        meta_dir = meta.parent if meta.suffix.lower() in {".json", ".npz"} or meta.is_file() else meta
        if meta_dir.parent.name == "processed":
            raw_sibling = meta_dir.parent.parent / "raw" / meta_dir.name
            if raw_sibling.exists() and raw_sibling.is_dir():
                return raw_sibling
            coco = PROJECT_ROOT / DEFAULT_COCO_REL
            if coco.exists() and coco.is_dir():
                return coco
        return meta_dir

    return DEFAULT_COCO_DIR


def filename_candidates(name: str) -> list[str]:
    """COCO ids are 12-digit padded; accept shorter variants like 0000000139.jpg."""
    raw = Path(_as_posix(name)).name
    if not raw:
        return []
    names = [raw]
    stem = Path(raw).stem
    suffix = Path(raw).suffix or ".jpg"
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        stripped = digits.lstrip("0") or "0"
        names.extend(
            [
                f"{digits.zfill(12)}{suffix}",
                f"{stripped.zfill(12)}{suffix}",
                f"{stripped}{suffix}",
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for item in names:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_project_path(path_str: str | Path | None) -> Path:
    """Resolve a stored path on this machine (Windows or Linux)."""
    if not path_str:
        return PROJECT_ROOT
    raw = str(path_str).strip()
    posix = _as_posix(raw)
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(PROJECT_ROOT / posix)
    lower = posix.lower()
    for marker in ("data/raw/", "data/processed/", "data/"):
        idx = lower.find(marker)
        if idx >= 0:
            candidates.append(PROJECT_ROOT / posix[idx:])
            break
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    if not Path(raw).is_absolute():
        return PROJECT_ROOT / posix
    idx = lower.find("data/")
    if idx >= 0:
        return PROJECT_ROOT / posix[idx:]
    return Path(raw)


def ensure_processed_dir(raw_path):
    raw_path = Path(raw_path)
    processed_path = raw_path.parent.parent / "processed" / raw_path.name
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path


def save_metadata(metadata, raw_path):
    processed_path = ensure_processed_dir(raw_path)
    output_path = processed_path / "metadata.json"
    for item in metadata:
        if item.get("image_path"):
            item["image_path"] = portable_relpath(item["image_path"])
            item.setdefault("filename", Path(item["image_path"]).name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return output_path


def load_metadata(metadata_path):
    path = resolve_project_path(metadata_path)
    if not path.exists():
        processed_path = (
            path.parent.parent / "processed" / path.name / "metadata.json"
        )
        if processed_path.exists():
            path = processed_path
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rewritten = False
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            original = item.get("image_path")
            if original:
                portable = portable_relpath(original)
                if portable != original:
                    item["image_path"] = portable
                    rewritten = True
                else:
                    item["image_path"] = portable
                item.setdefault("filename", Path(item["image_path"]).name)
    if rewritten:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError:
            pass
    return data


def get_unique_classes_counts(metadata):
    unique_classes = set()
    count_options = {}

    for item in metadata:
        counts = class_counts_of(item)
        for cls, count in counts.items():
            unique_classes.add(cls)
            count_options.setdefault(cls, set()).add(int(count))
        for det in detections_of(item):
            label = detection_label(det)
            if label:
                unique_classes.add(label)
                if "count" in det:
                    count_options.setdefault(label, set()).add(int(det["count"]))

    unique_classes = sorted(unique_classes)
    for cls in count_options:
        count_options[cls] = sorted(count_options[cls])
    return unique_classes, count_options


def _existing_file(path: Path) -> Path | None:
    try:
        if path.exists() and path.is_file():
            return path
    except OSError:
        return None
    return None


def _join_root_variants(root: Path, stored: str, names: list[str]) -> list[Path]:
    rel = stored_relative_path(stored)
    out: list[Path] = []
    if rel:
        out.append(root / rel)
        out.append(root / Path(rel).name)
    posix = _as_posix(stored)
    if posix and not Path(posix).is_absolute():
        out.append(root / posix)
    for name in names:
        out.append(root / name)
        out.append(root / "coco-val-2017-500" / name)
    return out


def resolve_image_path(
    image_path: str,
    extra_roots: list[Path] | None = None,
) -> Path | None:
    if not image_path:
        return None
    stored = str(image_path).strip()
    names = filename_candidates(stored)
    posix = _as_posix(stored)
    rel = stored_relative_path(stored)

    # A. Path exactly as stored, if it exists on this machine.
    found = _existing_file(Path(stored))
    if found:
        return found

    # B. Project root joined with the stored relative path (including data/ extracted
    # from an old absolute Windows path).
    if rel:
        found = _existing_file(PROJECT_ROOT / rel)
        if found:
            return found
    if posix and not Path(stored).is_absolute():
        found = _existing_file(PROJECT_ROOT / posix)
        if found:
            return found

    extra = [Path(item) for item in (extra_roots or []) if item]
    checked: set[str] = set()

    def _try(candidate: Path) -> Path | None:
        key = str(candidate)
        if key in checked:
            return None
        checked.add(key)
        return _existing_file(candidate)

    # C. Each collection image root + relative path and filename-only.
    for root in extra:
        for candidate in _join_root_variants(root, stored, names):
            found = _try(candidate)
            if found:
                return found

    # D. Configured / default COCO sample directory.
    coco_roots = [DEFAULT_COCO_DIR]
    configured = configured_image_root()
    if configured not in coco_roots:
        coco_roots.append(configured)
        coco_roots.append(configured / "coco-val-2017-500")
    for root in coco_roots:
        for name in names:
            found = _try(root / name)
            if found:
                return found
        if rel:
            found = _try(root / Path(rel).name)
            if found:
                return found

    # E. Other safe project-relative dataset roots.
    for root in image_lookup_roots(extra):
        for candidate in _join_root_variants(root, stored, names):
            found = _try(candidate)
            if found:
                return found
        try:
            if root.exists() and root.is_dir():
                for name in names:
                    matches = list(root.glob(f"**/{name}"))
                    if matches:
                        return matches[0]
        except OSError:
            continue
    return None


def image_resolution_diagnostics(
    image_path: str,
    extra_roots: list[Path] | None = None,
    filename: str | None = None,
) -> dict[str, str | bool | None]:
    stored = str(image_path or "").strip()
    names = filename_candidates(filename or stored)
    primary = filename or (names[0] if names else Path(stored).name)
    extra = [Path(item) for item in (extra_roots or []) if item]
    collection_root = extra[0] if extra else DEFAULT_COCO_DIR
    resolved = resolve_image_path(stored or primary, extra_roots=extra)
    exists = bool(resolved and _existing_file(resolved))
    return {
        "filename": primary,
        "stored_path": stored,
        "collection_root": str(collection_root),
        "exists": exists,
        "resolved_path": str(resolved) if resolved else None,
    }


def format_unresolved_image_diagnostics(
    image_path: str,
    extra_roots: list[Path] | None = None,
    filename: str | None = None,
) -> str:
    info = image_resolution_diagnostics(image_path, extra_roots=extra_roots, filename=filename)
    exists_txt = "yes" if info["exists"] else "no"
    return (
        f"filename: {info['filename']}\n"
        f"stored metadata path: {info['stored_path']}\n"
        f"attempted collection root: {info['collection_root']}\n"
        f"file exists: {exists_txt}"
    )


def collect_directory_images(directory: Path, extensions: list[str]) -> list[Path]:
    directory = resolve_project_path(directory)
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(sorted(directory.glob(f"*{ext}")))
    return paths
