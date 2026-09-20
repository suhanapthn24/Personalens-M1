"""Stage 1b: cleaning. Keeps maths symbols intact (NFC, not NFKC)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace

from .parsers import Segment

_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")  # "condi-\ntional" -> "conditional"
_SOFT_WRAP = re.compile(r"(?<![.!?:\n])\n(?=[a-z(])")  # line continues a sentence


def clean_text(text: str, kind: str = "prose", unwrap: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    text = unicodedata.normalize("NFC", text)
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = _CONTROL.sub("", text)

    if kind == "code":
        text = "\n".join(line.rstrip() for line in text.expandtabs(4).split("\n"))
        return re.sub(r"\n{3,}", "\n\n", text).strip("\n")

    if unwrap:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
        text = _SOFT_WRAP.sub(" ", text)
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{2,}", "\n", "\n".join(lines)).strip()


def strip_repeated_lines(
    segments: list[Segment], edge_lines: int = 2, min_pages: int = 4, ratio: float = 0.5
) -> list[Segment]:
    """Remove running headers/footers/page numbers from paged documents (PDF).

    A line is dropped only if it sits in the first/last `edge_lines` of a page AND it repeats on at
    least `ratio` of pages. Short lines (<= 40 chars) are compared with digits masked, so "Page 3"
    and "Page 4" match; longer lines must match exactly, so "Section 3. Foo" / "Section 4. Bar"
    style body text is never mistaken for a header. Pages with only a few lines are left alone.
    Run on raw page text, before clean_text() unwraps lines.
    """
    paged = [s for s in segments if s.page is not None]
    if len(paged) < min_pages:
        return segments

    def key(line: str) -> str:
        line = line.strip().lower()
        return re.sub(r"\d+", "#", line) if len(line) <= 40 else line

    def edge_indices(lines: list[str]) -> set[int]:
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        if len(nonblank) <= 2 * edge_lines:
            return set()
        return set(nonblank[:edge_lines] + nonblank[-edge_lines:])

    counts: Counter[str] = Counter()
    for s in paged:
        lines = s.text.split("\n")
        counts.update({key(lines[i]) for i in edge_indices(lines)})

    repeated = {k for k, c in counts.items() if c >= ratio * len(paged) and len(k) <= 120}
    if not repeated:
        return segments

    out = []
    for s in segments:
        if s.page is None:
            out.append(s)
            continue
        lines = s.text.split("\n")
        edges = edge_indices(lines)
        kept = [ln for i, ln in enumerate(lines) if not (i in edges and key(ln) in repeated)]
        out.append(replace(s, text="\n".join(kept)))
    return out
