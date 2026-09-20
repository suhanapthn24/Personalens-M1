from personalens.config import Settings
from personalens.ingestion.chunking import chunk_segments
from personalens.ingestion.cleaning import clean_text, strip_repeated_lines
from personalens.ingestion.parsers import Segment


def test_clean_text_pdf_artifacts():
    raw = "The condi-\ntional proba\u00adbility of \ufb01re\nis defined as\nP(A|B).\x00\x0cNext page"
    out = clean_text(raw, unwrap=True)
    assert "conditional" in out
    assert "fire" in out and "\ufb01" not in out
    assert "\x00" not in out
    assert "of fire is defined as P(A|B)." in out.replace("\n", " ") or "is defined as P(A|B)." in out


def test_clean_text_keeps_math_symbols():
    assert "x\u00b2" in clean_text("Variance is E[x\u00b2] - \u03bc\u00b2")
    assert "\u03bc" in clean_text("mean \u03bc")


def test_soft_wrap_only_joins_continuations():
    out = clean_text("First sentence ends.\nNew sentence starts\ncontinues here", unwrap=True)
    assert out == "First sentence ends.\nNew sentence starts continues here"


def _body(i):
    return "\n".join(f"Unique body line {j} for page {i} with words {i * 7 + j} and more filler text" for j in range(6))


def test_strip_repeated_header_footer():
    pages = [Segment(f"Lecture 3: Probability\n{_body(i)}\nPage {i}", page=i) for i in range(1, 7)]
    out = strip_repeated_lines(pages)
    for i, s in enumerate(out, start=1):
        assert "Lecture 3" not in s.text
        assert "Page " not in s.text.replace("for page", "")
        assert f"Unique body line 0 for page {i}" in s.text  # body untouched


def test_numbered_body_lines_are_not_mistaken_for_headers():
    pages = [Segment(f"Section {i}. Intro to topic {i} which is a long enough line to be body\n{_body(i)}\nend {i}", page=i) for i in range(1, 7)]
    out = strip_repeated_lines(pages)
    assert all(f"Section {i}." in s.text for i, s in enumerate(out, start=1))


def test_short_pages_left_alone():
    pages = [Segment(f"Header\nline a {i}\nline b {i}\nFooter", page=i) for i in range(1, 8)]
    assert strip_repeated_lines(pages) == pages


def test_strip_repeated_skips_short_docs():
    pages = [Segment("Header\nbody", page=i) for i in range(1, 3)]
    assert strip_repeated_lines(pages) == pages


def test_prose_chunk_size_overlap_and_coverage():
    cfg = Settings(chunk_max_words=40, chunk_overlap_words=10, chunk_min_words=5)
    sentences = [f"Sentence number {i} talks about topic {i} in some detail here." for i in range(30)]
    seg = Segment(" ".join(sentences), page=3, section="Topic")
    chunks = chunk_segments([seg], cfg)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= cfg.chunk_max_words for c in chunks)
    assert all(c.page == 3 and c.section == "Topic" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)  # nothing lost
    # each chunk starts by repeating the last sentence of the previous one
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.text.rsplit(". ", 1)[-1].rstrip(".")
        assert last_sentence in b.text


def test_overlong_sentence_is_hard_split():
    cfg = Settings(chunk_max_words=20, chunk_overlap_words=5)
    chunks = chunk_segments([Segment(" ".join(f"w{i}" for i in range(100)))], cfg)
    assert len(chunks) >= 5
    assert all(len(c.text.split()) <= 20 for c in chunks)


def test_tiny_segments_dropped():
    assert chunk_segments([Segment("Thank you!")], Settings()) == []


def test_line_structure_kept_for_bullets():
    seg = Segment("Key ideas\nPrior belief\nLikelihood of data\nPosterior update")
    (chunk,) = chunk_segments([seg], Settings())
    assert chunk.text.count("\n") == 3


def test_code_chunking_respects_definitions_and_decorators():
    body = "\n".join(f"    x{j} = {j}" for j in range(6))
    src = "import numpy as np\n\n" + "\n\n".join(
        f"@decorator\ndef func_{i}(a):\n{body}\n    return a" for i in range(8)
    )
    cfg = Settings(code_max_lines=25, code_overlap_lines=3)
    chunks = chunk_segments([Segment(src, kind="code")], cfg)
    assert len(chunks) > 1
    assert all(len(c.text.split("\n")) <= 25 for c in chunks)
    for c in chunks:
        for i, line in enumerate(c.text.split("\n")):
            if line.startswith("def "):
                assert i > 0 and c.text.split("\n")[i - 1].startswith("@")  # decorator kept with its def
    assert any(c.section and c.section.startswith(("def ", "@")) for c in chunks)


def test_long_code_block_is_windowed_with_overlap():
    src = "def big():\n" + "\n".join(f"    line_{i} = {i}" for i in range(100))
    cfg = Settings(code_max_lines=30, code_overlap_lines=4)
    chunks = chunk_segments([Segment(src, kind="code")], cfg)
    assert len(chunks) >= 4
    assert all(len(c.text.split("\n")) <= 30 for c in chunks)
    assert "line_99" in chunks[-1].text
