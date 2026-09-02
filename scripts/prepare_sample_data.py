"""Normalize bundled metadata paths and download the COCO val sample images."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import portable_relpath  # noqa: E402

META_PATH = ROOT / "data" / "processed" / "coco-val-2017-500" / "metadata.json"
RAW_DIR = ROOT / "data" / "raw" / "coco-val-2017-500"
COCO_URL = "https://images.cocodataset.org/val2017/{name}"
DOWNLOAD_WORKERS = 8


def normalize_metadata() -> list[str]:
    payload = json.loads(META_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for item in payload:
        original = item.get("image_path") or ""
        item["image_path"] = portable_relpath(original) if original else original
        name = Path(item["image_path"]).name if item.get("image_path") else ""
        if name:
            item["filename"] = name
            names.append(name)
    META_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return names


def _download_one(name: str) -> str:
    dest = RAW_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    url = COCO_URL.format(name=name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "yolo-image-search/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return "ok"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        print(f"failed {name}: {exc}")
        return "failed"


def download_images(names: list[str]) -> tuple[int, int, int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(names))
    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_one, name): name for name in unique}
        for i, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            if i % 25 == 0 or i == len(unique):
                print(
                    f"download progress {i}/{len(unique)} "
                    f"(new={ok} skipped={skipped} failed={failed})"
                )
    return ok, skipped, failed


def main() -> int:
    if not META_PATH.exists():
        print(f"metadata not found: {META_PATH}")
        return 1
    names = normalize_metadata()
    print(f"normalized {len(names)} metadata paths -> {META_PATH}")
    ok, skipped, failed = download_images(names)
    print(f"images: downloaded={ok} already_present={skipped} failed={failed} dir={RAW_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
