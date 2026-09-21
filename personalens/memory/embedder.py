"""Stage 2 (embeddings). Swap embedders without touching the rest of M1."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 array (n, dim), L2-normalised (so inner product == cosine)."""
        ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (384-d) by default. Model loads lazily on first encode()."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        max_seq_length: int = 256,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.dim = 384 if "MiniLM-L6" in model_name else 0  # resolved on first load otherwise

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
            self.dim = int(self._model.get_embedding_dimension())
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs, dtype="float32")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests/CI and offline demos only:
    texts sharing words get similar vectors, but it has no real semantic understanding."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_name = "hash-embedder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(out / norms, dtype="float32")
