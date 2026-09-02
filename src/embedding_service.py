"""CLIP image/text embeddings for semantic search.

Uses OpenCLIP ViT-B-32. This is visual-language similarity, not YOLO attribute labels.
Falls back cleanly if the package or weights are unavailable.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
from PIL import Image


CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class EmbeddingService:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.available = False
        self.error = None
        self._load()

    def _load(self) -> None:
        try:
            import open_clip
        except ImportError:
            self.error = (
                "Semantic search requires the open-clip-torch package. "
                "Install it with: pip install open-clip-torch"
            )
            return
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL, pretrained=CLIP_PRETRAINED
            )
            model = model.to(self.device)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
            self.available = True
        except Exception as exc:
            self.error = f"Could not load the CLIP semantic model: {exc}"

    @torch.no_grad()
    def encode_image(self, image: Image.Image | str | Path) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self.error or "Semantic model unavailable.")
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        vec = features.detach().cpu().float().numpy()[0]
        return _l2_normalize(vec.astype(np.float32))

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self.error or "Semantic model unavailable.")
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        vec = features.detach().cpu().float().numpy()[0]
        return _l2_normalize(vec.astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def save_embeddings(path: str | Path, ids: list[str], matrix: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=matrix.astype(np.float32))
    return path


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    if not path:
        return {}
    from src.utils import resolve_project_path

    path = resolve_project_path(path)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    vectors = data["vectors"]
    return {image_id: vectors[i] for i, image_id in enumerate(ids)}
