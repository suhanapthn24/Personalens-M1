"""POST /documents (plan section 6.3) plus list / delete / search helpers for M1."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import Settings
from ..ingestion.pipeline import IngestionPipeline, UnsupportedFileType
from ..schemas import DocumentRecord, IngestResult, SearchHit


def create_router(pipeline: IngestionPipeline, settings: Settings | None = None) -> APIRouter:
    settings = settings or pipeline.settings
    memory = pipeline.memory
    max_bytes = settings.max_upload_mb * 1024 * 1024
    router = APIRouter(tags=["memory"])

    # Plain `def` (not async): FastAPI runs it in a worker thread, so parsing/embedding don't block
    # the event loop. Moving to BackgroundTasks later means returning a "processing" status here.
    @router.post("/documents", response_model=IngestResult)
    def upload_document(
        file: UploadFile = File(...),
        user_id: str = Form(...),
        source_type: str | None = Form(None, description="Override: pdf|slide|note|code|task"),
        kind: str = Form("quiz_answer", description="For .csv: quiz_answer | task_grade"),
        column_map: str | None = Form(None, description="For .csv: JSON mapping canonical field -> CSV column"),
    ):
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / f"upload{suffix}"
            size = 0
            with dest.open("wb") as out:
                while block := file.file.read(1 << 20):
                    size += len(block)
                    if size > max_bytes:
                        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
                    out.write(block)
            try:
                if suffix == ".csv":
                    cmap = json.loads(column_map) if column_map else None
                    return pipeline.ingest_quiz_csv(
                        dest, user_id, event_type=kind, column_map=cmap, filename=file.filename
                    )
                return pipeline.ingest_file(dest, user_id, source_type=source_type, filename=file.filename)
            except UnsupportedFileType as e:
                raise HTTPException(415, str(e))
            except ValueError as e:  # bad CSV, bad source_type, encrypted PDF, invalid JSON, ...
                raise HTTPException(422, str(e))

    @router.get("/documents", response_model=list[DocumentRecord])
    def list_documents(user_id: str):
        return memory.list_documents(user_id)

    @router.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, user_id: str):
        if not memory.delete_document(user_id, doc_id):
            raise HTTPException(404, "Document not found for this user")
        return {"deleted": doc_id}

    # Dev/debug aid so M1 retrieval quality can be eyeballed. Module 2 owns the real hybrid retrieval.
    @router.get("/memory/search", response_model=list[SearchHit])
    def search(user_id: str, q: str, k: int = 5):
        return memory.search(q, user_id, k=k)

    return router
