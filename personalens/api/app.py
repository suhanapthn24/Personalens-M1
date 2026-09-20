"""Standalone app for M1: `uvicorn --factory personalens.api.app:create_app`.

For the integrated backend, Module 4 should just mount the router:
    app.include_router(create_router(pipeline))
"""
from __future__ import annotations

from fastapi import FastAPI

from ..config import Settings
from ..ingestion.pipeline import IngestionPipeline
from ..memory.embedder import Embedder, SentenceTransformerEmbedder
from ..memory.vector_store import VectorMemory
from .documents import create_router


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or Settings()
    embedder = embedder or SentenceTransformerEmbedder(
        settings.embedding_model, max_seq_length=settings.max_seq_length
    )
    pipeline = IngestionPipeline(VectorMemory(settings.data_dir, embedder), settings)
    app = FastAPI(title="PersonaLens: M1 ingestion & vector memory")
    app.include_router(create_router(pipeline, settings))
    return app
