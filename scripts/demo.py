"""Ingest a file and run a query.
   python scripts/demo.py --user suhana --file UnitI.pdf --query "conditional probability"
Add --offline to use the hashing embedder (no model download; retrieval is keyword-ish only).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalens import HashEmbedder, IngestionPipeline, SentenceTransformerEmbedder, Settings, VectorMemory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--file", action="append", default=[], help="repeatable")
ap.add_argument("--query")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--data-dir", default="data")
ap.add_argument("--offline", action="store_true")
args = ap.parse_args()

settings = Settings(data_dir=Path(args.data_dir))
embedder = HashEmbedder() if args.offline else SentenceTransformerEmbedder(settings.embedding_model)
pipeline = IngestionPipeline(VectorMemory(settings.data_dir, embedder), settings)

for f in args.file:
    r = pipeline.ingest_file(f, args.user)
    print(f"{f}: {r.status}, {r.n_chunks} chunks", *r.warnings, sep="\n  " if r.warnings else "")

if args.query:
    for hit in pipeline.memory.search(args.query, args.user, k=args.k):
        c = hit.chunk
        loc = f"p.{c.page}" if c.page else c.section or ""
        print(f"\n[{hit.score:.3f}] chunk {c.chunk_id} ({c.source_type} {loc})\n{c.text[:300]}")
