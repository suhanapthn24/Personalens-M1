from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("PERSONALENS_DATA_DIR", "data")))

    # Embeddings (plan: all-MiniLM-L6-v2, 384-d, runs locally)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 256  # MiniLM silently truncates beyond this many word-pieces

    # Prose chunking. ~150 words is roughly 200 word-pieces, safely under max_seq_length.
    chunk_max_words: int = 150
    chunk_overlap_words: int = 30
    chunk_min_words: int = 5

    # Code chunking (code tokenises into more word-pieces per line than prose)
    code_max_lines: int = 30
    code_overlap_lines: int = 4

    max_upload_mb: int = 50
    retrieval_min_score: float = 0.35
