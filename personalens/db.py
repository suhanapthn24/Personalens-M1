"""SQLite: source of truth for documents, chunks and raw events.

The FAISS index is a derived artifact and can always be rebuilt from `chunks`.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    filename    TEXT NOT NULL,
    source_type TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    file_hash   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

-- AUTOINCREMENT: ids are never reused, so "chunk_id:12" in an evidence trace stays valid.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    source_type TEXT NOT NULL,
    page        INTEGER,
    section     TEXT,
    chunk_index INTEGER NOT NULL,
    word_count  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);

-- No FK to documents: events must outlive a deleted document (they are evidence).
CREATE TABLE IF NOT EXISTS raw_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    doc_id     TEXT,
    chunk_id   INTEGER,
    concept    TEXT,
    item_id    TEXT,
    outcome    REAL,
    value      REAL,
    payload    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON raw_events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_concept   ON raw_events(user_id, concept);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def close(self) -> None:
        self.conn.close()
