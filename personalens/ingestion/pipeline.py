"""Stages 1-3 (write path) glued together: parse -> clean -> chunk -> embed -> store + index."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import get_args

from ..config import Settings
from ..memory.vector_store import VectorMemory
from ..schemas import Chunk, ChunkSourceType, DocumentRecord, IngestResult, utcnow
from .chunking import chunk_segments
from .cleaning import clean_text, strip_repeated_lines
from .events import EventStore, load_quiz_csv
from .parsers import CODE_EXTS, parse_file

EXT_TO_SOURCE: dict[str, str] = {
    ".pdf": "pdf",
    ".pptx": "slide",
    ".docx": "note",
    ".txt": "note",
    ".md": "note",
    ".markdown": "note",
    **{e: "code" for e in CODE_EXTS},
}


class UnsupportedFileType(ValueError):
    pass


def _doc_id(path: Path, user_id: str) -> str:
    """Stable id from user + file bytes, so re-uploading the same file is detected."""
    h = hashlib.sha256()
    h.update(user_id.encode())
    h.update(b"\0")
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


class IngestionPipeline:
    def __init__(self, memory: VectorMemory, settings: Settings | None = None):
        self.memory = memory
        self.settings = settings or Settings()
        self.events = EventStore(memory.db)

    def ingest_file(
        self,
        path: str | Path,
        user_id: str,
        *,
        source_type: str | None = None,
        filename: str | None = None,
    ) -> IngestResult:
        """Ingest a PDF / PPTX / DOCX / note / code file. `source_type="task"` marks a task brief."""
        path = Path(path)
        filename = filename or path.name
        ext = Path(filename).suffix.lower()

        if ext == ".csv":
            raise UnsupportedFileType("CSV files are quiz/task logs: use ingest_quiz_csv().")
        if ext not in EXT_TO_SOURCE:
            raise UnsupportedFileType(f"Unsupported file type {ext!r}. Supported: {sorted(EXT_TO_SOURCE)}")
        if source_type is not None and source_type not in get_args(ChunkSourceType):
            raise ValueError(f"source_type must be one of {get_args(ChunkSourceType)}")
        if path.stat().st_size > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"File is larger than {self.settings.max_upload_mb} MB")

        doc_id = _doc_id(path, user_id)
        if self.memory.has_document(doc_id):
            return IngestResult(doc_id=doc_id, filename=filename, status="duplicate")

        parsed = parse_file(path, ext)
        segments = strip_repeated_lines(parsed.segments) if ext == ".pdf" else parsed.segments
        cleaned = []
        for seg in segments:
            text = clean_text(seg.text, seg.kind, seg.unwrap)
            if text:
                cleaned.append(replace(seg, text=text))

        raw_chunks = chunk_segments(cleaned, self.settings)
        if not raw_chunks:
            return IngestResult(
                doc_id=doc_id, filename=filename, status="empty",
                warnings=[*parsed.warnings, "No usable text was extracted."],
            )

        src: str = source_type or EXT_TO_SOURCE[ext]
        now = utcnow()
        chunks = [
            Chunk(
                doc_id=doc_id, user_id=user_id, text=rc.text, uploaded_at=now, source_type=src,
                page=rc.page, section=rc.section, chunk_index=i, word_count=len(rc.text.split()),
            )
            for i, rc in enumerate(raw_chunks)
        ]
        record = DocumentRecord(
            doc_id=doc_id, user_id=user_id, filename=filename, source_type=src, uploaded_at=now,
        )
        stored = self.memory.add_document(record, chunks)
        return IngestResult(
            doc_id=doc_id, filename=filename, status="ingested", n_chunks=len(stored), warnings=parsed.warnings
        )

    def ingest_quiz_csv(
        self,
        path: str | Path,
        user_id: str,
        *,
        event_type: str = "quiz_answer",
        column_map: dict[str, str] | None = None,
        filename: str | None = None,
    ) -> IngestResult:
        """Quiz results (event_type="quiz_answer") or graded-task logs ("task_grade") -> raw_events."""
        path = Path(path)
        filename = filename or path.name
        events, warnings = load_quiz_csv(
            path, user_id, event_type=event_type, column_map=column_map, source_name=filename
        )
        n = self.events.add_many(events)
        return IngestResult(filename=filename, status="events_ingested", n_events=n, warnings=warnings)
