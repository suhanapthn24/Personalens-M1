"""Stage 1d: quiz / graded-task logs and interaction events -> raw_events (evidence candidates).

Module 3 (Stage 4) reads these. This module only stores them; it does not weigh or interpret them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..db import Database
from ..schemas import EventType, RawEvent, utcnow

_TRUE = {"1", "true", "t", "yes", "y", "correct", "right", "pass", "passed"}
_FALSE = {"0", "false", "f", "no", "n", "incorrect", "wrong", "fail", "failed"}

# canonical field -> default CSV column name (case-insensitive). Override with column_map.
DEFAULT_COLUMNS = {
    "timestamp": "timestamp",
    "item_id": "question_id",
    "concept": "concept",
    "outcome": "correct",
    "score": "score",
    "max_score": "max_score",
}


def _parse_outcome(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in _TRUE:
        return 1.0
    if s in _FALSE:
        return 0.0
    try:
        x = float(s)
    except ValueError:
        return None
    return x if 0.0 <= x <= 1.0 else None


def _score_to_outcome(score, max_score) -> float | None:
    try:
        s = float(score)
        m = float(max_score) if max_score is not None and not pd.isna(max_score) else 1.0
    except (TypeError, ValueError):
        return None
    if m <= 0 or s < 0 or s > m:
        return None
    return s / m


def _clean(v) -> str | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def load_quiz_csv(
    path: str | Path,
    user_id: str,
    *,
    event_type: EventType = "quiz_answer",
    column_map: dict[str, str] | None = None,
    source_name: str | None = None,
) -> tuple[list[RawEvent], list[str]]:
    """Parse a quiz / graded-task CSV into RawEvents. Returns (events, warnings).

    Needs an outcome column (correct/incorrect) or a score column (+ optional max_score).
    Rows with no usable outcome are skipped. Rows with no parseable timestamp get the import
    time and payload["timestamp_imputed"] = True, so Stage 4 can discount them (recency matters).
    """
    if event_type not in ("quiz_answer", "task_grade"):
        raise ValueError("event_type must be 'quiz_answer' or 'task_grade' for CSV imports")

    cols = {**DEFAULT_COLUMNS, **(column_map or {})}
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def col(name: str) -> str | None:
        wanted = cols[name].strip()
        return wanted if wanted in df.columns else lower.get(wanted.lower())

    c_ts, c_item, c_concept, c_out, c_score, c_max = (
        col(n) for n in ("timestamp", "item_id", "concept", "outcome", "score", "max_score")
    )
    if not c_out and not c_score:
        raise ValueError(
            f"CSV needs an outcome column (default 'correct') or a score column. "
            f"Found columns: {list(df.columns)}. Use column_map to point at them."
        )

    warnings: list[str] = []
    if not c_ts:
        warnings.append("No timestamp column found: every row was stamped with the import time.")
        ts_parsed = pd.Series([pd.NaT] * len(df))
    else:
        ts_parsed = pd.to_datetime(df[c_ts], errors="coerce", utc=True, format="mixed")

    now = utcnow()
    events: list[RawEvent] = []
    skipped = imputed = 0
    for pos, (_, row) in enumerate(df.iterrows()):
        outcome = _parse_outcome(row[c_out]) if c_out else None
        if outcome is None and c_score:
            outcome = _score_to_outcome(row[c_score], row[c_max] if c_max else None)
        if outcome is None:
            skipped += 1
            continue

        ts = ts_parsed.iloc[pos]
        payload: dict = {"source": source_name or Path(path).name, "row": pos + 1}
        if pd.isna(ts):
            ts_dt: datetime = now
            payload["timestamp_imputed"] = True
            imputed += 1
        else:
            ts_dt = ts.to_pydatetime()

        events.append(
            RawEvent(
                user_id=user_id,
                event_type=event_type,
                timestamp=ts_dt,
                concept=_clean(row[c_concept]) if c_concept else None,
                item_id=_clean(row[c_item]) if c_item else None,
                outcome=outcome,
                payload=payload,
            )
        )

    if skipped:
        warnings.append(f"Skipped {skipped} row(s) with no usable outcome/score.")
    if imputed and c_ts:
        warnings.append(f"{imputed} row(s) had no parseable timestamp and were stamped with the import time.")
    return events, warnings


class EventStore:
    def __init__(self, db: Database):
        self.db = db

    def add_many(self, events: Iterable[RawEvent]) -> int:
        rows = [
            (
                e.user_id, e.event_type, e.timestamp.isoformat(), e.doc_id, e.chunk_id,
                e.concept, e.item_id, e.outcome, e.value, json.dumps(e.payload, default=str),
            )
            for e in events
        ]
        with self.db.lock, self.db.conn:
            self.db.conn.executemany(
                "INSERT INTO raw_events(user_id,event_type,timestamp,doc_id,chunk_id,concept,"
                "item_id,outcome,value,payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def add(self, event: RawEvent) -> int:
        """Log one event (e.g. a page open or a question asked). Returns its event_id."""
        with self.db.lock, self.db.conn:
            cur = self.db.conn.execute(
                "INSERT INTO raw_events(user_id,event_type,timestamp,doc_id,chunk_id,concept,"
                "item_id,outcome,value,payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event.user_id, event.event_type, event.timestamp.isoformat(), event.doc_id,
                    event.chunk_id, event.concept, event.item_id, event.outcome, event.value,
                    json.dumps(event.payload, default=str),
                ),
            )
            return int(cur.lastrowid)

    def list(
        self,
        user_id: str,
        *,
        event_types: list[str] | None = None,
        concept: str | None = None,
        since: datetime | None = None,
    ) -> list[RawEvent]:
        sql, args = "SELECT * FROM raw_events WHERE user_id = ?", [user_id]
        if event_types:
            sql += f" AND event_type IN ({','.join('?' * len(event_types))})"
            args += list(event_types)
        if concept:
            sql += " AND concept = ?"
            args.append(concept)
        if since:
            sql += " AND timestamp >= ?"
            args.append(since.isoformat())
        sql += " ORDER BY timestamp, event_id"
        with self.db.lock:
            rows = self.db.conn.execute(sql, args).fetchall()
        return [
            RawEvent(
                event_id=r["event_id"], user_id=r["user_id"], event_type=r["event_type"],
                timestamp=datetime.fromisoformat(r["timestamp"]), doc_id=r["doc_id"],
                chunk_id=r["chunk_id"], concept=r["concept"], item_id=r["item_id"],
                outcome=r["outcome"], value=r["value"], payload=json.loads(r["payload"]),
            )
            for r in rows
        ]
