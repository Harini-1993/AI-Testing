from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from config import get_config
from app.pipeline.chunker import Chunk

if TYPE_CHECKING:
    import numpy as np


EmbeddingBackend = Literal["sentence_transformers", "openai"]

class Embedder:
    """Generate and cache embeddings for text chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path = ".cache",
    ) -> None:
        self.model_name = model_name or get_config(validate=False).embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._detect_backend(self.model_name)
        self._model: Any | None = None

    def embed(self, chunks: list[Chunk] | list[str]) -> list[np.ndarray]:
        """Embed chunks or raw strings.

        Chunk inputs use the local cache. Raw string inputs are supported for
        older callers and are embedded directly without metadata-based caching.
        """
        if chunks and isinstance(chunks[0], str):
            return self.embed_texts(chunks)

        embeddings: list[np.ndarray | None] = []
        missing: list[tuple[int, Chunk]] = []

        for index, chunk in enumerate(chunks):
            cached = self._load_cached_embedding(chunk)
            if cached is None:
                embeddings.append(None)
                missing.append((index, chunk))
            else:
                embeddings.append(cached)

        if missing:
            generated = self.embed_texts([chunk.text for _, chunk in missing])
            for (index, chunk), embedding in zip(missing, generated, strict=True):
                embeddings[index] = embedding
                self._save_cached_embedding(chunk, embedding)

        return [embedding for embedding in embeddings if embedding is not None]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one ad hoc text string without using the chunk cache."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed raw texts with the configured backend."""
        np = _numpy()

        if self.backend == "sentence_transformers":
            model = self._load_sentence_transformer()
            encoded = model.encode(list(texts), convert_to_numpy=True)
            return [np.asarray(vector, dtype=np.float32) for vector in encoded]

        client = self._load_openai_client()
        response = client.embeddings.create(model=self.model_name, input=list(texts))
        return [
            np.asarray(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    def _detect_backend(self, model_name: str) -> EmbeddingBackend:
        if model_name.startswith("sentence-transformers/"):
            return "sentence_transformers"

        if model_name.startswith("text-embedding-"):
            return "openai"

        raise ValueError(f"Unsupported embedding model backend: {model_name}")

    def _load_sentence_transformer(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def _load_openai_client(self) -> Any:
        if self._model is None:
            from openai import OpenAI

            self._model = OpenAI()

        return self._model

    def _cache_path(self, chunk: Chunk) -> Path:
        metadata = chunk.metadata
        cache_key = "|".join(
            [
                str(metadata.get("source_file", "")),
                str(metadata.get("version_label", "")),
                str(metadata.get("chunk_index", "")),
                self.model_name,
            ]
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_cached_embedding(self, chunk: Chunk) -> np.ndarray | None:
        np = _numpy()
        cache_path = self._cache_path(chunk)
        if not cache_path.exists():
            return None

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("text_hash") != _text_hash(chunk.text):
            return None

        return np.asarray(payload["embedding"], dtype=np.float32)

    def _save_cached_embedding(self, chunk: Chunk, embedding: np.ndarray) -> None:
        payload = {
            "text_hash": _text_hash(chunk.text),
            "embedding": embedding.astype(float).tolist(),
        }
        self._cache_path(chunk).write_text(json.dumps(payload), encoding="utf-8")


def load_embedding_model(
    model_name: str | None = None,
) -> Any:
    """Load a sentence-transformers model for legacy callers."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name or get_config(validate=False).embedding_model)


def generate_embeddings(
    texts: Sequence[str],
    model: Any,
) -> list[list[float]]:
    """Generate embeddings for a sequence of text chunks for legacy callers."""
    embeddings = model.encode(list(texts), convert_to_numpy=True)
    return embeddings.tolist()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numpy() -> Any:
    import numpy as np

    return np
