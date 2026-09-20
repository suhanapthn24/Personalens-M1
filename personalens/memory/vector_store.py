"""Stage 3 (write + read path for vectors): FAISS index per user + SQLite chunk store.

* SQLite is the source of truth; each user's FAISS file is derived and self-heals: if it is
  missing or its size disagrees with SQLite on load, it is rebuilt by re-embedding the chunks.
* chunk_id (SQLite AUTOINCREMENT) doubles as the FAISS id, so search hits map straight to rows.
* One index per user gives hard isolation with no post-filtering misses. IndexFlatIP over
  normalised vectors is exact cosine search, which is plenty for 1-2 subjects.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import faiss
import numpy as np

from ..db import Database
from ..schemas import Chunk, DocumentRecord, SearchHit
from .embedder import Embedder

log = logging.getLogger(__name__)

_CHUNK_COLS = "chunk_id, doc_id, user_id, text, uploaded_at, source_type, page, section, chunk_index, word_count"


def _row_to_chunk(r) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"], doc_id=r["doc_id"], user_id=r["user_id"], text=r["text"],
        uploaded_at=r["uploaded_at"], source_type=r["source_type"], page=r["page"],
        section=r["section"], chunk_index=r["chunk_index"], word_count=r["word_count"],
    )


class VectorMemory:
    def __init__(self, data_dir: str | Path, embedder: Embedder, db: Database | None = None):
        self.data_dir = Path(data_dir)
        self.index_dir = self.data_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or Database(self.data_dir / "personalens.db")
        self.embedder = embedder
        self.lock = self.db.lock
        self._indexes: dict[str, faiss.Index] = {}

    # ------------------------------------------------------------ index plumbing
    def _index_path(self, user_id: str) -> Path:
        return self.index_dir / f"{quote(user_id, safe='')}.faiss"

    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))

    def _save_index(self, user_id: str) -> None:
        path = self._index_path(user_id)
        tmp = path.with_suffix(".faiss.tmp")
        faiss.write_index(self._indexes[user_id], str(tmp))
        os.replace(tmp, path)  # atomic: never leaves a half-written index

    def _count_chunks(self, user_id: str) -> int:
        return self.db.conn.execute("SELECT COUNT(*) FROM chunks WHERE user_id = ?", (user_id,)).fetchone()[0]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.ascontiguousarray(self.embedder.encode(texts), dtype="float32")
        if vecs.ndim != 2 or vecs.shape[0] != len(texts) or vecs.shape[1] != self.embedder.dim:
            raise ValueError(f"Embedder returned shape {vecs.shape}, expected ({len(texts)}, {self.embedder.dim})")
        return vecs

    def _get_index(self, user_id: str) -> faiss.Index:
        with self.lock:
            idx = self._indexes.get(user_id)
            if idx is not None:
                return idx
            path = self._index_path(user_id)
            n_db = self._count_chunks(user_id)
            if path.exists():
                idx = faiss.read_index(str(path))
                if idx.d == self.embedder.dim and idx.ntotal == n_db:
                    self._indexes[user_id] = idx
                    return idx
                log.warning("FAISS index for %r out of sync with SQLite (%s vs %s); rebuilding", user_id, idx.ntotal, n_db)
            elif n_db:
                log.warning("FAISS index for %r missing but %s chunks exist; rebuilding", user_id, n_db)
            if n_db:
                return self.rebuild_index(user_id)
            self._indexes[user_id] = self._new_index()
            return self._indexes[user_id]

    def rebuild_index(self, user_id: str, batch_size: int = 256) -> faiss.Index:
        """Re-embed every chunk for a user from SQLite. Use after changing the embedding model."""
        with self.lock:
            rows = self.db.conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE user_id = ? ORDER BY chunk_id", (user_id,)
            ).fetchall()
            idx = self._new_index()
            for s in range(0, len(rows), batch_size):
                chunks = [_row_to_chunk(r) for r in rows[s : s + batch_size]]
                vecs = self._embed([c.embed_text for c in chunks])
                idx.add_with_ids(vecs, np.asarray([c.chunk_id for c in chunks], dtype="int64"))
            self._indexes[user_id] = idx
            self._save_index(user_id)
            return idx

    # ------------------------------------------------------------ write path
    def has_document(self, doc_id: str) -> bool:
        with self.lock:
            return self.db.conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is not None

    def add_document(self, doc: DocumentRecord, chunks: list[Chunk]) -> list[Chunk]:
        """Store a document + its chunks and index their embeddings. All-or-nothing."""
        if not chunks:
            raise ValueError("add_document needs at least one chunk")
        vectors = self._embed([c.embed_text for c in chunks])  # slow part first, outside the txn

        with self.lock:
            idx = self._get_index(doc.user_id)
            ids: list[int] = []
            added = False
            try:
                with self.db.conn:  # one transaction: commit on success, roll back on error
                    self.db.conn.execute(
                        "INSERT INTO documents(doc_id,user_id,filename,source_type,uploaded_at,n_chunks) VALUES (?,?,?,?,?,?)",
                        (doc.doc_id, doc.user_id, doc.filename, doc.source_type, doc.uploaded_at.isoformat(), len(chunks)),
                    )
                    for c in chunks:
                        cur = self.db.conn.execute(
                            "INSERT INTO chunks(doc_id,user_id,text,uploaded_at,source_type,page,section,chunk_index,word_count) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (c.doc_id, c.user_id, c.text, c.uploaded_at.isoformat(), c.source_type,
                             c.page, c.section, c.chunk_index, c.word_count),
                        )
                        ids.append(int(cur.lastrowid))
                    idx.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
                    added = True
            except Exception:
                if added:  # commit itself failed after the index add
                    idx.remove_ids(np.asarray(ids, dtype="int64"))
                raise
            self._save_index(doc.user_id)
        return [c.model_copy(update={"chunk_id": i}) for c, i in zip(chunks, ids)]

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        with self.lock:
            rows = self.db.conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? AND user_id = ?", (doc_id, user_id)
            ).fetchall()
            owned = self.db.conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ? AND user_id = ?", (doc_id, user_id)
            ).fetchone()
            if not owned:
                return False
            idx = self._get_index(user_id)
            with self.db.conn:
                self.db.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))  # cascades to chunks
            if rows:
                idx.remove_ids(np.asarray([r["chunk_id"] for r in rows], dtype="int64"))
            self._save_index(user_id)
            return True

    # ------------------------------------------------------------ read path
    def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        *,
        min_score: float = 0.35,
        doc_ids: Sequence[str] | None = None,
        source_types: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Top-k chunks by cosine similarity. This is the vector half of Module 2's hybrid retrieval."""
        with self.lock:
            idx = self._get_index(user_id)
            if idx.ntotal == 0 or k <= 0:
                return []
            filtered = bool(doc_ids or source_types)
            fetch = idx.ntotal if filtered else min(k, idx.ntotal)
            scores, ids = idx.search(self._embed([query]), fetch)

            ranked = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
            hits: list[SearchHit] = []
            for start in range(0, len(ranked), 200):  # batch to stay under SQLite's variable limit
                batch = ranked[start : start + 200]
                by_id = self.get_chunks([i for i, _ in batch])
                for cid, score in batch:
                    if score < min_score:
                        continue
                    c = by_id.get(cid)
                    if c is None or c.user_id != user_id:
                        continue
                    if doc_ids and c.doc_id not in doc_ids:
                        continue
                    if source_types and c.source_type not in source_types:
                        continue
                    hits.append(SearchHit(chunk=c, score=score))
                    if len(hits) >= k:
                        return hits
            return hits

    def get_chunks(self, chunk_ids: Sequence[int]) -> dict[int, Chunk]:
        out: dict[int, Chunk] = {}
        ids = list(chunk_ids)
        with self.lock:
            for s in range(0, len(ids), 500):
                part = ids[s : s + 500]
                rows = self.db.conn.execute(
                    f"SELECT {_CHUNK_COLS} FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})", part
                ).fetchall()
                out.update({r["chunk_id"]: _row_to_chunk(r) for r in rows})
        return out

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks of a document in reading order. Module 2 uses this for concept extraction."""
        with self.lock:
            rows = self.db.conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def list_documents(self, user_id: str) -> list[DocumentRecord]:
        with self.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at", (user_id,)
            ).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]
