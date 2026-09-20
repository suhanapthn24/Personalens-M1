"""Data contracts shared with the other modules.

Chunk mirrors the shape in the plan (Stage 1) plus page/section/chunk_index.
RawEvent is what Module 3 (Varun) reads as "evidence candidates" for Stage 4.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Quizzes and interaction events are NOT chunks; they go to raw_events.
ChunkSourceType = Literal["pdf", "slide", "note", "code", "task"]

EventType = Literal[
    "quiz_answer", "task_grade", "self_report", "open", "question", "time_on_page", "revisit"
]

# Evidence families from Stage 4 of the plan.
EVENT_FAMILY: dict[str, str] = {
    "quiz_answer": "performance",
    "task_grade": "performance",
    "self_report": "self_report",
    "open": "behaviour",
    "question": "behaviour",
    "time_on_page": "behaviour",
    "revisit": "behaviour",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Chunk(BaseModel):
    chunk_id: int | None = None  # assigned by SQLite (AUTOINCREMENT, never reused); also the FAISS id
    doc_id: str
    user_id: str
    text: str
    uploaded_at: datetime
    source_type: ChunkSourceType
    page: int | None = None  # PDF page or slide number, 1-based
    section: str | None = None  # nearest heading / slide title / code signature
    chunk_index: int  # position within the document
    word_count: int

    @property
    def embed_text(self) -> str:
        """Text actually embedded: the section heading is prepended when the chunk lacks it."""
        if self.section and not self.text.startswith(self.section):
            return f"{self.section}\n{self.text}"
        return self.text


class DocumentRecord(BaseModel):
    doc_id: str
    user_id: str
    filename: str
    source_type: ChunkSourceType
    uploaded_at: datetime
    n_chunks: int = 0


class RawEvent(BaseModel):
    event_id: int | None = None
    user_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=utcnow)
    doc_id: str | None = None
    chunk_id: int | None = None
    concept: str | None = None  # raw label as given; M2 maps it to a canonical concept_id
    item_id: str | None = None  # question / task id
    outcome: float | None = Field(default=None, ge=0.0, le=1.0)  # 1 = correct, partial credit allowed
    value: float | None = None  # seconds for time_on_page, 1-5 for self_report, etc.
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    @property
    def family(self) -> str:
        return EVENT_FAMILY[self.event_type]


class IngestResult(BaseModel):
    doc_id: str | None = None
    filename: str
    status: Literal["ingested", "duplicate", "empty", "events_ingested"]
    n_chunks: int = 0
    n_events: int = 0
    warnings: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity (embeddings are L2-normalised)
