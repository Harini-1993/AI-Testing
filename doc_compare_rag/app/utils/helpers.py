from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline.chunker import Chunk


def load_environment(env_path: str | Path = ".env") -> None:
    """Load environment variables from a dotenv file."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)


def format_diff_as_html(diff_lines: list[str]) -> str:
    """Format unified diff lines as HTML spans for Streamlit markdown."""
    html_lines: list[str] = []

    for line in diff_lines:
        escaped = escape(line.rstrip("\n"))
        if line.startswith("+") and not line.startswith("+++"):
            html_lines.append(f'<span style="color:green">{escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            html_lines.append(f'<span style="color:red">{escaped}</span>')
        else:
            html_lines.append(escaped)

    return "<br>".join(html_lines)


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Return text shortened to max_chars with an ellipsis when needed."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    if max_chars <= 3:
        return "." * max_chars

    return f"{cleaned[: max_chars - 3].rstrip()}..."


def build_rag_prompt(question: str, context_chunks: list["Chunk"]) -> str:
    """Build a QA prompt from a question and retrieved context chunks."""
    passages = []

    for index, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.metadata
        source = metadata.get("source_file", "unknown source")
        version = metadata.get("version_label", "unknown version")
        page = metadata.get("page_number", "unknown page")
        section = metadata.get("section_title", "")
        heading = f"[{index}] Source: {source}, Version: {version}, Page: {page}"
        if section:
            heading += f", Section: {section}"
        passages.append(f"{heading}\n{chunk.text}")

    context = "\n\n".join(passages) if passages else "No context retrieved."
    return (
        "Answer the question using only the context passages below. "
        "If the answer is not present, say that the documents do not provide it.\n\n"
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Answer:"
    )
