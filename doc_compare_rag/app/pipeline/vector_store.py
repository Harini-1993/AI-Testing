from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from config import get_config
from app.pipeline.chunker import Chunk
from app.pipeline.embedder import Embedder

if TYPE_CHECKING:
    import numpy as np


VectorStoreBackend = Literal["faiss", "chroma"]

class VectorStoreManager:
    """Persist and query version-scoped vector indexes."""

    def __init__(
        self,
        backend: VectorStoreBackend | None = None,
        store_type: VectorStoreBackend | None = None,
        persist_dir: str | Path = "data",
        embedder: Embedder | None = None,
    ) -> None:
        configured_backend = backend or store_type or get_config(validate=False).vector_store
        if configured_backend not in {"faiss", "chroma"}:
            raise ValueError(f"Unsupported vector store backend: {configured_backend}")

        self.backend: VectorStoreBackend = configured_backend
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()

    def index(
        self,
        version_label: str,
        chunks: list[Chunk],
        embeddings: list[np.ndarray],
    ) -> None:
        """Index chunks and embeddings in a namespace for one document version."""
        if not chunks:
            raise ValueError("chunks cannot be empty")

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        if self.backend == "faiss":
            self._index_faiss(version_label, chunks, embeddings)
            return

        self._index_chroma(version_label, chunks, embeddings)

    def search(
        self,
        query: str,
        version_label: str,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Search one document version and return chunks ranked by cosine score."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedder.embed_text(query)

        if self.backend == "faiss":
            return self._search_faiss(version_label, query_embedding, top_k)

        return self._search_chroma(version_label, query_embedding, top_k)

    def search_all_versions(
        self,
        query: str,
        top_k_per_version: int = 3,
    ) -> dict[str, list[Chunk]]:
        """Search every persisted version namespace."""
        if top_k_per_version <= 0:
            raise ValueError("top_k_per_version must be positive")

        return {
            version_label: self.search(query, version_label, top_k_per_version)
            for version_label in self._known_versions()
        }

    def _index_faiss(
        self,
        version_label: str,
        chunks: list[Chunk],
        embeddings: list[np.ndarray],
    ) -> None:
        import faiss

        matrix = _normalise_matrix(embeddings)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        faiss.write_index(index, str(self._faiss_index_path(version_label)))
        self._faiss_chunks_path(version_label).write_text(
            json.dumps([_chunk_to_payload(chunk) for chunk in chunks], indent=2),
            encoding="utf-8",
        )

    def _search_faiss(
        self,
        version_label: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[Chunk]:
        import faiss

        index_path = self._faiss_index_path(version_label)
        chunks_path = self._faiss_chunks_path(version_label)
        if not index_path.exists() or not chunks_path.exists():
            return []

        index = faiss.read_index(str(index_path))
        chunks = [
            _chunk_from_payload(payload)
            for payload in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        query = _normalise_matrix([query_embedding])
        scores, indices = index.search(query, min(top_k, len(chunks)))

        results: list[Chunk] = []
        for score, chunk_index in zip(scores[0], indices[0], strict=True):
            if chunk_index < 0:
                continue
            results.append(chunks[int(chunk_index)].with_score(float(score)))

        return results

    def _index_chroma(
        self,
        version_label: str,
        chunks: list[Chunk],
        embeddings: list[np.ndarray],
    ) -> None:
        client = self._chroma_client()
        collection_name = _collection_name(version_label)

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine", "version_label": version_label},
        )
        collection.add(
            ids=[_chunk_id(version_label, chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=[embedding.astype(float).tolist() for embedding in embeddings],
        )

    def _search_chroma(
        self,
        version_label: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[Chunk]:
        client = self._chroma_client()
        collection_name = _collection_name(version_label)

        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return []

        response = collection.query(
            query_embeddings=[query_embedding.astype(float).tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        return [
            Chunk(text=text, metadata=dict(metadata)).with_score(1.0 - float(distance))
            for text, metadata, distance in zip(
                documents, metadatas, distances, strict=True
            )
        ]

    def _known_versions(self) -> list[str]:
        if self.backend == "faiss":
            return sorted(path.stem for path in self.persist_dir.glob("*.faiss"))

        client = self._chroma_client()
        versions: list[str] = []
        for collection in client.list_collections():
            name = collection.name if hasattr(collection, "name") else str(collection)
            if name.startswith("version_"):
                versions.append(name.removeprefix("version_"))
        return sorted(versions)

    def _chroma_client(self) -> Any:
        import chromadb

        return chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))

    def _faiss_index_path(self, version_label: str) -> Path:
        return self.persist_dir / f"{version_label}.faiss"

    def _faiss_chunks_path(self, version_label: str) -> Path:
        return self.persist_dir / f"{version_label}.chunks.json"


def build_vector_store(
    texts: list[str],
    embeddings: list[list[float]],
    document_version: str,
    backend: VectorStoreBackend = "faiss",
    persist_dir: str | Path = "data",
) -> VectorStoreManager:
    """Build and persist a vector store for legacy callers."""
    chunks = [
        Chunk(
            text=text,
            metadata={
                "source_file": "",
                "version_label": document_version,
                "page_number": 0,
                "section_title": "",
                "file_type": "",
                "chunk_index": index,
                "total_chunks": len(texts),
            },
        )
        for index, text in enumerate(texts)
    ]
    manager = VectorStoreManager(backend=backend, persist_dir=persist_dir)
    np = _numpy()
    manager.index(document_version, chunks, [np.asarray(vector) for vector in embeddings])
    return manager


def _normalise_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    np = _numpy()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "metadata": chunk.metadata,
        "score": chunk.score,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        text=payload["text"],
        metadata=dict(payload["metadata"]),
        score=payload.get("score"),
    )


def _chunk_id(version_label: str, chunk: Chunk) -> str:
    chunk_index = chunk.metadata.get("chunk_index", 0)
    return f"{version_label}-{chunk_index}"


def _collection_name(version_label: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in version_label
    )
    return f"version_{safe}"


def _numpy() -> Any:
    import numpy as np

    return np
