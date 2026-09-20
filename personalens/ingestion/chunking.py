"""Stage 1c: chunking.

Prose: sentence-aware packing to <= max_words with sentence-level overlap. Chunks never cross a
segment boundary, so page / slide / section metadata stays exact.
Code: split at top-level definitions, packed to <= max_lines, long blocks windowed with overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings
from .parsers import Segment

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CODE_BOUNDARY = re.compile(
    r"^(?:async\s+def|def|class|function|export\s+(?:default\s+)?(?:async\s+)?(?:function|class)"
    r"|public|private|protected|static|fn|func)\b"
)


@dataclass
class RawChunk:
    text: str
    page: int | None
    section: str | None


@dataclass
class _Unit:
    text: str
    words: int
    nl: bool  # followed by a line break in the source


# ---------------------------------------------------------------- prose
def _units(text: str, max_words: int) -> list[_Unit]:
    units: list[_Unit] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences = [s for s in _SENT_SPLIT.split(line) if s.strip()]
        for j, sent in enumerate(sentences):
            words = sent.split()
            for k in range(0, len(words), max_words):  # hard-split over-long sentences
                piece = words[k : k + max_words]
                units.append(_Unit(" ".join(piece), len(piece), nl=False))
            if j == len(sentences) - 1 and units:
                units[-1].nl = True
    return units


def _pack(units: list[_Unit], max_words: int, overlap_words: int) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_words = 0
    new_units = 0  # units in `cur` that are not just overlap seed
    for u in units:
        if cur and cur_words + u.words > max_words:
            groups.append(cur)
            seed: list[_Unit] = []
            w = 0
            for prev in reversed(cur):
                fits = w + prev.words <= overlap_words
                # always carry at least the last sentence if it is reasonably short
                first_ok = not seed and prev.words <= max_words // 2
                if not (fits or first_ok):
                    break
                seed.insert(0, prev)
                w += prev.words
            while seed and w + u.words > max_words:
                w -= seed.pop(0).words
            cur, cur_words, new_units = seed, w, 0
        cur.append(u)
        cur_words += u.words
        new_units += 1
    if cur and new_units:
        groups.append(cur)
    return groups


def _render(units: list[_Unit]) -> str:
    out = []
    for i, u in enumerate(units):
        out.append(u.text)
        if i < len(units) - 1:
            out.append("\n" if u.nl else " ")
    return "".join(out)


def _chunk_prose(seg: Segment, cfg: Settings) -> list[RawChunk]:
    groups = _pack(_units(seg.text, cfg.chunk_max_words), cfg.chunk_max_words, cfg.chunk_overlap_words)
    chunks = []
    for g in groups:
        if sum(u.words for u in g) >= cfg.chunk_min_words:
            chunks.append(RawChunk(_render(g), seg.page, seg.section))
    return chunks


# ---------------------------------------------------------------- code
def _code_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        is_start = bool(_CODE_BOUNDARY.match(line)) or line.startswith("@")
        attached = bool(cur) and cur[-1].startswith("@")  # decorator stays with its def
        if is_start and cur and not attached:
            blocks.append(cur)
            cur = []
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def _signature(lines: list[str]) -> str | None:
    first = next((ln for ln in lines if ln.strip() and not ln.startswith("@")), "")
    return first.strip()[:80] if _CODE_BOUNDARY.match(first) else None


def _chunk_code(seg: Segment, cfg: Settings) -> list[RawChunk]:
    max_lines, overlap = cfg.code_max_lines, min(cfg.code_overlap_lines, cfg.code_max_lines - 1)
    groups: list[tuple[list[str], str | None]] = []
    cur: list[str] = []

    def emit(lines: list[str], section: str | None = None) -> None:
        if any(ln.strip() for ln in lines):
            groups.append((lines, section or _signature(lines)))

    for block in _code_blocks(seg.text.split("\n")):
        if len(block) > max_lines:
            if cur:
                emit(cur)
                cur = []
            step = max_lines - overlap
            sig = _signature(block)  # every window of a long function keeps its name
            for s in range(0, len(block), step):
                emit(block[s : s + max_lines], sig)
                if s + max_lines >= len(block):
                    break
            continue
        if cur and len(cur) + len(block) > max_lines:
            emit(cur)
            cur = []
        cur = cur + block
    if cur:
        emit(cur)

    return [RawChunk("\n".join(g).strip("\n"), seg.page, section) for g, section in groups]


# ---------------------------------------------------------------- entry point
def chunk_segments(segments: list[Segment], cfg: Settings) -> list[RawChunk]:
    out: list[RawChunk] = []
    for seg in segments:
        out.extend(_chunk_code(seg, cfg) if seg.kind == "code" else _chunk_prose(seg, cfg))
    return out
