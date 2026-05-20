import difflib
from dataclasses import dataclass
from typing import Literal

from app.ingestion.document import Document


SectionStatus = Literal["added", "removed", "modified", "unchanged"]


@dataclass
class SectionDiff:
    """A line-level diff for one aligned document section."""

    section_title: str
    status: SectionStatus
    diff_lines: list[str]
    text_a: str
    text_b: str


@dataclass
class ComparisonResult:
    """Structured comparison output for two document versions."""

    version_a_label: str
    version_b_label: str
    sections: list[SectionDiff]
    summary_stats: dict[str, int]


class DocumentDiffer:
    """Align two document versions by section title and produce unified diffs."""

    def compare(
        self,
        version_a: list[Document],
        version_b: list[Document],
    ) -> ComparisonResult:
        """Compare two lists of normalized documents."""
        version_a_label = _version_label(version_a, "version_a")
        version_b_label = _version_label(version_b, "version_b")
        sections: list[SectionDiff] = []

        for section_a, section_b in _align_sections(version_a, version_b):
            text_a = section_a.text if section_a else ""
            text_b = section_b.text if section_b else ""
            title = _aligned_title(section_a, section_b)
            status = _section_status(section_a, section_b, text_a, text_b)
            diff_lines = _unified_diff_lines(
                text_a,
                text_b,
                fromfile=version_a_label,
                tofile=version_b_label,
            )

            sections.append(
                SectionDiff(
                    section_title=title,
                    status=status,
                    diff_lines=diff_lines,
                    text_a=text_a,
                    text_b=text_b,
                )
            )

        return ComparisonResult(
            version_a_label=version_a_label,
            version_b_label=version_b_label,
            sections=sections,
            summary_stats=_summary_stats(sections),
        )


def diff_text(
    original_text: str,
    revised_text: str,
    fromfile: str = "original",
    tofile: str = "revised",
) -> str:
    """Return a unified diff between two text versions."""
    original_lines = original_text.splitlines(keepends=True)
    revised_lines = revised_text.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        revised_lines,
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff)


def _align_sections(
    version_a: list[Document],
    version_b: list[Document],
) -> list[tuple[Document | None, Document | None]]:
    grouped_a = _group_by_section(version_a)
    grouped_b = _group_by_section(version_b)

    titled_a = {title: document for title, document in grouped_a if title}
    titled_b = {title: document for title, document in grouped_b if title}
    untitled_a = [document for title, document in grouped_a if not title]
    untitled_b = [document for title, document in grouped_b if not title]
    shared_titles = [title for title in titled_a if title in titled_b]
    aligned: list[tuple[Document | None, Document | None]] = [
        (titled_a[title], titled_b[title]) for title in shared_titles
    ]

    removed_titles = [title for title in titled_a if title not in titled_b]
    added_titles = [title for title in titled_b if title not in titled_a]

    aligned.extend((titled_a[title], None) for title in removed_titles)
    aligned.extend((None, titled_b[title]) for title in added_titles)

    max_length = max(len(untitled_a), len(untitled_b))
    for index in range(max_length):
        aligned.append(
            (
                untitled_a[index] if index < len(untitled_a) else None,
                untitled_b[index] if index < len(untitled_b) else None,
            )
        )

    return aligned


def _group_by_section(documents: list[Document]) -> list[tuple[str, Document]]:
    grouped: list[tuple[str, Document]] = []
    buckets: dict[str, list[str]] = {}
    metadata_by_title: dict[str, dict] = {}

    for index, document in enumerate(documents):
        raw_title = str(document.metadata.get("section_title", "")).strip()
        title = raw_title or f"__position_{index}"
        buckets.setdefault(title, []).append(document.text)
        metadata_by_title.setdefault(title, document.metadata)

    for title, texts in buckets.items():
        display_title = "" if title.startswith("__position_") else title
        grouped.append(
            (
                display_title,
                Document(
                    text="\n".join(text for text in texts if text.strip()),
                    metadata={**metadata_by_title[title], "section_title": display_title},
                ),
            )
        )

    return grouped


def _aligned_title(
    section_a: Document | None,
    section_b: Document | None,
) -> str:
    if section_a:
        title_a = str(section_a.metadata.get("section_title", "")).strip()
        if title_a:
            return title_a

    if section_b:
        title_b = str(section_b.metadata.get("section_title", "")).strip()
        if title_b:
            return title_b

    return ""


def _section_status(
    section_a: Document | None,
    section_b: Document | None,
    text_a: str,
    text_b: str,
) -> SectionStatus:
    if section_a is None:
        return "added"

    if section_b is None:
        return "removed"

    if text_a == text_b:
        return "unchanged"

    return "modified"


def _unified_diff_lines(
    text_a: str,
    text_b: str,
    *,
    fromfile: str,
    tofile: str,
) -> list[str]:
    return list(
        difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _summary_stats(sections: list[SectionDiff]) -> dict[str, int]:
    stats = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0, "total": 0}
    for section in sections:
        stats[section.status] += 1
        stats["total"] += 1
    return stats


def _version_label(documents: list[Document], fallback: str) -> str:
    for document in documents:
        label = str(document.metadata.get("version_label", "")).strip()
        if label:
            return label

    return fallback
