import pytest

from personalens import HashEmbedder, IngestionPipeline, VectorMemory
from personalens.ingestion.pipeline import UnsupportedFileType

from conftest import BAYES, GRADIENT


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_end_to_end_ingest_and_search(pipeline, tmp_path):
    r1 = pipeline.ingest_file(_write(tmp_path, "bayes.md", f"# Bayes\n{BAYES}"), "u1")
    r2 = pipeline.ingest_file(_write(tmp_path, "gd.md", f"# Optimisation\n{GRADIENT}"), "u1")
    assert r1.status == r2.status == "ingested" and r1.n_chunks >= 1

    hits = pipeline.memory.search("posterior likelihood prior Bayes", "u1", k=2)
    assert hits[0].chunk.doc_id == r1.doc_id
    assert hits[0].score > hits[1].score
    assert hits[0].chunk.chunk_id is not None and hits[0].chunk.source_type == "note"

    hits = pipeline.memory.search("learning rate step size", "u1", k=1)
    assert hits[0].chunk.doc_id == r2.doc_id


def test_duplicate_upload_detected(pipeline, tmp_path):
    p = _write(tmp_path, "a.md", BAYES)
    assert pipeline.ingest_file(p, "u1").status == "ingested"
    dup = pipeline.ingest_file(p, "u1")
    assert dup.status == "duplicate"
    assert pipeline.memory._count_chunks("u1") == len(pipeline.memory.get_document_chunks(dup.doc_id))
    # the same file for a different user is a separate document
    assert pipeline.ingest_file(p, "u2").status == "ingested"


def test_user_isolation(pipeline, tmp_path):
    pipeline.ingest_file(_write(tmp_path, "a.md", BAYES), "alice")
    assert pipeline.memory.search("Bayes Theorem", "bob") == []
    assert pipeline.memory.search("Bayes Theorem", "alice")


def test_filters(pipeline, tmp_path):
    a = pipeline.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    b = pipeline.ingest_file(_write(tmp_path, "b.py", "def bayes(prior, likelihood):\n    return prior * likelihood\n"), "u1")
    hits = pipeline.memory.search("bayes prior likelihood", "u1", k=5, source_types=["code"])
    assert hits and all(h.chunk.source_type == "code" and h.chunk.doc_id == b.doc_id for h in hits)
    hits = pipeline.memory.search("bayes", "u1", k=5, doc_ids=[a.doc_id])
    assert all(h.chunk.doc_id == a.doc_id for h in hits)


def test_source_type_override_and_validation(pipeline, tmp_path):
    p = _write(tmp_path, "brief.md", BAYES)
    r = pipeline.ingest_file(p, "u1", source_type="task")
    assert pipeline.memory.get_document_chunks(r.doc_id)[0].source_type == "task"
    with pytest.raises(ValueError):
        pipeline.ingest_file(_write(tmp_path, "x.md", GRADIENT), "u1", source_type="quiz")


def test_unsupported_and_empty(pipeline, tmp_path):
    with pytest.raises(UnsupportedFileType):
        pipeline.ingest_file(_write(tmp_path, "x.exe", "hi"), "u1")
    with pytest.raises(UnsupportedFileType):
        pipeline.ingest_file(_write(tmp_path, "q.csv", "a,b"), "u1")
    res = pipeline.ingest_file(_write(tmp_path, "e.txt", "   \n  "), "u1")
    assert res.status == "empty" and res.warnings


def test_persistence_across_restart(settings, tmp_path):
    p1 = IngestionPipeline(VectorMemory(settings.data_dir, HashEmbedder()), settings)
    r = p1.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    p1.memory.db.close()

    p2 = IngestionPipeline(VectorMemory(settings.data_dir, HashEmbedder()), settings)
    hits = p2.memory.search("Bayes Theorem", "u1")
    assert hits and hits[0].chunk.doc_id == r.doc_id


def test_index_self_heals_when_file_missing_or_stale(settings, tmp_path):
    mem = VectorMemory(settings.data_dir, HashEmbedder())
    pipe = IngestionPipeline(mem, settings)
    pipe.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    pipe.ingest_file(_write(tmp_path, "b.md", GRADIENT), "u1")
    expected = len(mem.search("anything", "u1", k=50))
    mem.db.close()
    for f in (settings.data_dir / "indexes").glob("*.faiss"):
        f.unlink()

    mem2 = VectorMemory(settings.data_dir, HashEmbedder())
    assert len(mem2.search("Bayes", "u1", k=50)) == expected
    assert list((settings.data_dir / "indexes").glob("*.faiss"))  # rebuilt file written back


def test_delete_document_removes_vectors_and_rows(pipeline, tmp_path):
    a = pipeline.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    b = pipeline.ingest_file(_write(tmp_path, "b.md", GRADIENT), "u1")
    assert not pipeline.memory.delete_document("intruder", a.doc_id)
    assert pipeline.memory.delete_document("u1", a.doc_id)
    assert not pipeline.memory.has_document(a.doc_id)
    hits = pipeline.memory.search("Bayes Theorem posterior", "u1", k=10)
    assert all(h.chunk.doc_id == b.doc_id for h in hits)
    assert [d.doc_id for d in pipeline.memory.list_documents("u1")] == [b.doc_id]


def test_chunk_ids_never_reused_after_delete(pipeline, tmp_path):
    a = pipeline.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    old_ids = {c.chunk_id for c in pipeline.memory.get_document_chunks(a.doc_id)}
    pipeline.memory.delete_document("u1", a.doc_id)
    b = pipeline.ingest_file(_write(tmp_path, "b.md", GRADIENT), "u1")
    new_ids = {c.chunk_id for c in pipeline.memory.get_document_chunks(b.doc_id)}
    assert not (old_ids & new_ids)


def test_failed_embedding_leaves_nothing_behind(settings, tmp_path):
    class Boom(HashEmbedder):
        def encode(self, texts):
            raise RuntimeError("model unavailable")

    pipe = IngestionPipeline(VectorMemory(settings.data_dir, Boom()), settings)
    with pytest.raises(RuntimeError):
        pipe.ingest_file(_write(tmp_path, "a.md", BAYES), "u1")
    assert pipe.memory.list_documents("u1") == []
    assert pipe.memory._count_chunks("u1") == 0


def test_pdf_pipeline_keeps_pages_and_strips_footers(pipeline, tmp_path):
    import pymupdf

    doc = pymupdf.open()
    for i in range(1, 7):
        page = doc.new_page()
        page.insert_text((72, 40), "Probability Lecture 3", fontsize=9)
        name = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"][i - 1]
        body = " ".join(f"The {name} method shows {name} behaviour in situation {w} clearly." for w in
                        ["one", "two", "three", "four", "five", "six", "seven", "eight"])
        page.insert_textbox(pymupdf.Rect(72, 80, 520, 700), body, fontsize=10)
        page.insert_text((280, 800), f"Page {i}", fontsize=9)
    p = tmp_path / "lec.pdf"
    doc.save(p)

    res = pipeline.ingest_file(p, "u1")
    chunks = pipeline.memory.get_document_chunks(res.doc_id)
    assert res.status == "ingested" and {c.page for c in chunks} == set(range(1, 7))
    assert all("Probability Lecture 3" not in c.text for c in chunks)
    assert all(c.source_type == "pdf" for c in chunks)


def test_search_empty_and_bad_k(pipeline):
    assert pipeline.memory.search("x", "nobody") == []
    assert pipeline.memory.search("x", "nobody", k=0) == []
