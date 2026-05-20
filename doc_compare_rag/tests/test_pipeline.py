from pathlib import Path

import pytest

from app.ingestion.document import Document
from app.pipeline.chunker import Chunk, TextChunker, chunk_text
from app.pipeline.embedder import Embedder
from app.pipeline.vector_store import VectorStoreManager


def test_chunk_text_uses_overlap() -> None:
    text = "one two three four five six"

    assert chunk_text(text, chunk_size=3, overlap=1) == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("hello world", chunk_size=2, overlap=2)


def test_text_chunker_respects_chunk_size_and_overlap() -> None:
    document = _document("abcdefghijklmnopqrstuvwxyz0123456789")
    chunker = TextChunker(chunk_size=12, chunk_overlap=4)

    chunks = chunker.split([document])

    assert all(len(chunk.text) <= 12 for chunk in chunks)
    assert chunks[0].text[-4:] == chunks[1].text[:4]


def test_text_chunker_preserves_parent_metadata() -> None:
    document = _document("First sentence. Second sentence. Third sentence.")

    chunks = TextChunker(chunk_size=20, chunk_overlap=5).split([document])

    assert chunks
    for chunk in chunks:
        for key, value in document.metadata.items():
            assert chunk.metadata[key] == value
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_embedder_output_shape_and_determinism(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    chunks = [
        _chunk("alpha", chunk_index=0),
        _chunk("beta", chunk_index=1),
    ]
    embedder = DeterministicEmbedder(cache_dir=tmp_path)

    first = embedder.embed(chunks)
    second = embedder.embed(chunks)

    assert np.vstack(first).shape == (2, 3)
    assert np.array_equal(np.vstack(first), np.vstack(second))


def test_vector_store_index_and_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    storage: dict[str, list[Chunk]] = {}
    manager = VectorStoreManager(
        backend="faiss",
        persist_dir=tmp_path,
        embedder=FakeSearchEmbedder(),
    )
    monkeypatch.setattr(
        manager,
        "_index_faiss",
        lambda version_label, chunks, embeddings: storage.update(
            {version_label: chunks}
        ),
    )
    monkeypatch.setattr(
        manager,
        "_search_faiss",
        lambda version_label, query_embedding, top_k: [
            chunk.with_score(1.0 - index * 0.1)
            for index, chunk in enumerate(storage[version_label][:top_k])
        ],
    )

    chunks = [_chunk("alpha", 0), _chunk("beta", 1), _chunk("gamma", 2)]
    embeddings = [np.array([index, index + 1], dtype=np.float32) for index in range(3)]

    manager.index("v1", chunks, embeddings)
    results = manager.search("alpha", "v1", top_k=2)

    assert len(results) == 2
    assert results[0].text == "alpha"
    assert results[0].metadata["source_file"] == "sample.pdf"
    assert results[0].score is not None


def test_vector_store_search_all_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    storage = {
        "v1": [_chunk("alpha v1", 0, version_label="v1")],
        "v2": [_chunk("alpha v2", 0, version_label="v2")],
    }
    manager = VectorStoreManager(
        backend="faiss",
        persist_dir=tmp_path,
        embedder=FakeSearchEmbedder(),
    )
    monkeypatch.setattr(manager, "_known_versions", lambda: sorted(storage))
    monkeypatch.setattr(
        manager,
        "_search_faiss",
        lambda version_label, query_embedding, top_k: [
            chunk.with_score(0.9) for chunk in storage[version_label][:top_k]
        ],
    )

    results = manager.search_all_versions("alpha", top_k_per_version=1)

    assert set(results) == {"v1", "v2"}
    assert results["v1"][0].text == "alpha v1"
    assert results["v2"][0].metadata["version_label"] == "v2"


class DeterministicEmbedder(Embedder):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__(
            model_name="sentence-transformers/test-double",
            cache_dir=cache_dir,
        )

    def embed_texts(self, texts):
        np = pytest.importorskip("numpy")
        return [
            np.array(
                [
                    len(text),
                    sum(ord(character) for character in text) % 101,
                    text.count("a"),
                ],
                dtype=np.float32,
            )
            for text in texts
        ]


class FakeSearchEmbedder:
    def embed_text(self, query: str):
        np = pytest.importorskip("numpy")
        return np.array([1.0, 0.0], dtype=np.float32)


def _document(text: str) -> Document:
    return Document(
        text=text,
        metadata={
            "source_file": "sample.pdf",
            "version_label": "v1",
            "page_number": 1,
            "section_title": "Intro",
            "file_type": "pdf_text",
        },
    )


def _chunk(
    text: str,
    chunk_index: int,
    version_label: str = "v1",
) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "source_file": "sample.pdf",
            "version_label": version_label,
            "page_number": 1,
            "section_title": "Intro",
            "file_type": "pdf_text",
            "chunk_index": chunk_index,
            "total_chunks": 3,
        },
    )
