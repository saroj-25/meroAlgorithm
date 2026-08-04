"""Chunking for ingesting *your own* course material (Section 3.2).

Paper heuristic: 250-450 tokens per chunk with a 50-token overlap.  Splitting on
paragraph boundaries first keeps a worked example or a proof inside one chunk,
which matters more for retrieval quality than hitting the token target exactly.
"""

from __future__ import annotations

from typing import Iterator, List


def word_tokens(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, min_tokens: int = 250, max_tokens: int = 450,
               overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks on paragraph boundaries where possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            chunks.append("\n\n".join(buffer).strip())

    for paragraph in paragraphs:
        buffer_len = sum(len(word_tokens(p)) for p in buffer)
        para_len = len(word_tokens(paragraph))

        if buffer_len + para_len <= max_tokens:
            buffer.append(paragraph)
            continue
        if buffer_len >= min_tokens:
            flush()
            # Carry the last `overlap` tokens forward so context is not cut mid-idea.
            tail = " ".join(word_tokens("\n\n".join(buffer))[-overlap:])
            buffer = [tail, paragraph] if tail else [paragraph]
            continue
        # A single oversized paragraph: split it on the token grid.
        words = word_tokens(paragraph)
        start = 0
        while start < len(words):
            piece = words[start:start + max_tokens]
            if buffer:
                buffer.append(" ".join(piece))
                flush()
                buffer = []
            else:
                chunks.append(" ".join(piece))
            start += max_tokens - overlap
    flush()
    return [c for c in chunks if c]


def iter_chunks(text: str, **kwargs) -> Iterator[str]:
    yield from chunk_text(text, **kwargs)
