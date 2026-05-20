import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "anthropic"
    vector_store: str = "faiss"
    chunk_size: int = 1_000
    chunk_overlap: int = 200
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    @classmethod
    def from_env(cls, env_path: str | Path = ".env", validate: bool = False) -> "Config":
        """Load configuration from .env and the current process environment."""
        _load_dotenv(env_path)
        config = cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider).lower(),
            vector_store=os.getenv("VECTOR_STORE", cls.vector_store).lower(),
            chunk_size=_int_env("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", cls.chunk_overlap),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

        if validate:
            config.validate()

        return config

    def validate(self, require_llm_key: bool = True) -> None:
        """Raise clear errors for invalid or missing required configuration."""
        errors: list[str] = []

        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be a positive integer")
        if self.chunk_overlap < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.vector_store not in {"faiss", "chroma"}:
            errors.append("VECTOR_STORE must be either 'faiss' or 'chroma'")
        if not (
            self.embedding_model.startswith("sentence-transformers/")
            or self.embedding_model.startswith("text-embedding-")
        ):
            errors.append(
                "EMBEDDING_MODEL must start with 'sentence-transformers/' "
                "or 'text-embedding-'"
            )
        if self.llm_provider not in {"anthropic", "openai"}:
            errors.append("LLM_PROVIDER must be either 'anthropic' or 'openai'")
        if self.embedding_model.startswith("text-embedding-") and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for OpenAI embeddings")
        if require_llm_key and self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if (
            require_llm_key
            and self.llm_provider == "anthropic"
            and not self.anthropic_api_key
        ):
            errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))


def get_config(validate: bool = False, require_llm_key: bool = True) -> Config:
    """Return application configuration, optionally validating required keys."""
    config = Config.from_env(validate=False)
    if validate:
        config.validate(require_llm_key=require_llm_key)
    return config


def _load_dotenv(env_path: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(dotenv_path=env_path)


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
