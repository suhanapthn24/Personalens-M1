# PersonaLens — Module 1: Data Ingestion & Vector Memory

Covers pipeline stages 1, 2 (embeddings) and 3 (FAISS) from the implementation plan.

```
file ─► parse ─► clean ─► chunk ─► embed (MiniLM, 384-d) ─► SQLite (chunks) + FAISS (vectors)
csv  ─► raw_events (quiz / task / interaction evidence for Module 3)
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 42 tests, no model download needed

python scripts/demo.py --user suhana --file UnitI.pdf --query "conditional probability"
uvicorn --factory personalens.api.app:create_app --reload   # POST /documents, GET /memory/search
```

First real run downloads all-MiniLM-L6-v2 (~90 MB). Tests use `HashEmbedder` (no semantics) so CI stays offline.

## Layout

| Path | What it does |
|---|---|
| `ingestion/parsers.py` | PDF (PyMuPDF), DOCX, PPTX (incl. speaker notes), MD/TXT, code, `.ipynb` → `Segment`s with page/section |
| `ingestion/cleaning.py` | Ligatures, soft hyphens, PDF line un-wrapping, running header/footer removal. Uses NFC (not NFKC) so `x²`, `μ` survive |
| `ingestion/chunking.py` | Prose: sentence-aware, ≤150 words, 1-sentence overlap, never crosses a page/slide. Code: splits at top-level defs, ≤30 lines |
| `ingestion/events.py` | Quiz/task CSV → `RawEvent`; `EventStore` for interaction events |
| `ingestion/pipeline.py` | `IngestionPipeline.ingest_file()` / `.ingest_quiz_csv()` |
| `memory/embedder.py` | `SentenceTransformerEmbedder` (swappable via the `Embedder` protocol) |
| `memory/vector_store.py` | `VectorMemory`: per-user FAISS index + SQLite; `search`, `get_chunks`, `get_document_chunks`, `delete_document` |
| `api/documents.py` | `POST /documents`, `GET /documents`, `DELETE /documents/{id}`, `GET /memory/search` |

## Hand-off contracts

**Module 2 (Ananya)**
- `memory.get_document_chunks(doc_id)` → ordered `Chunk`s for concept/triple extraction. Keep `chunk_id` as provenance.
- `memory.search(query, user_id, k, doc_ids=None, source_types=None)` → `SearchHit(chunk, score)` is the vector half of hybrid retrieval.
- `chunk_id` is an int, never reused, and is also the FAISS id.

**Module 3 (Varun)** reads the `raw_events` table (or `EventStore.list(...)`):
`event_type` ∈ quiz_answer, task_grade, self_report, open, question, time_on_page, revisit;
`outcome` ∈ [0,1] for performance events; `concept` is the *raw label* (M2 maps it to a canonical id).
Rows with `payload.timestamp_imputed = true` had no parseable timestamp, so discount them in recency.

**Module 4 (Sammed)**: `app.include_router(create_router(pipeline))` mounts the endpoints.