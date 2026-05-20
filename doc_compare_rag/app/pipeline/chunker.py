from dataclasses import dataclass, replace

from config import get_config
from app.ingestion.document import Document


@dataclass
class Chunk:
    """A searchable text fragment with inherited document metadata."""

    text: str
    metadata: dict
    score: float | None = None

    def with_score(self, score: float) -> "Chunk":
        """Return a copy of the chunk annotated with a similarity score."""
        return replace(self, score=score, metadata={**self.metadata, "score": score})


class TextChunker:
    """Split ingestion documents into overlapping, metadata-preserving chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        config = get_config(validate=False)
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        _validate_chunk_settings(self.chunk_size, self.chunk_overlap)

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks using a sliding character window."""
        chunks: list[Chunk] = []

        for document in documents:
            chunk_texts = self._split_text(document.text)
            total_chunks = len(chunk_texts)

            for chunk_index, text in enumerate(chunk_texts):
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                        },
                    )
                )

        return chunks

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """Split one text string with metadata into chunks.

        This convenience wrapper keeps older orchestration code compatible with
        the Document-based split API.
        """
        return self.split([Document(text=text, metadata=metadata)])

    def _split_text(self, text: str) -> list[str]:
        cleaned_text = " ".join(text.split())
        if not cleaned_text:
            return []

        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned_text)

        while start < text_length:
            hard_end = min(start + self.chunk_size, text_length)
            end = _sentence_boundary(cleaned_text, start, hard_end)
            chunk = cleaned_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)
            while start < text_length and cleaned_text[start].isspace():
                start += 1

        return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks.

    Kept for existing callers and tests. New ingestion code should use
    TextChunker.split with Document objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


def _sentence_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)

    boundary = text.rfind(". ", start, hard_end)
    if boundary > start:
        return boundary + 1

    return hard_end


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
