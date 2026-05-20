import pytest

from app.comparison.differ import DocumentDiffer, diff_text
from app.comparison.summariser import DiffSummariser
from app.ingestion.document import Document


def test_diff_text_returns_unified_diff() -> None:
    diff = diff_text("alpha\nbeta\n", "alpha\ngamma\n")

    assert "--- original" in diff
    assert "+++ revised" in diff
    assert "-beta" in diff
    assert "+gamma" in diff


def test_document_differ_detects_modified_added_and_removed_sections() -> None:
    result = DocumentDiffer().compare(_version_a(), _version_b())
    statuses = {section.section_title: section.status for section in result.sections}

    assert statuses["Shared"] == "modified"
    assert statuses["Removed"] == "removed"
    assert statuses["Added"] == "added"
    assert result.summary_stats == {
        "added": 1,
        "removed": 1,
        "modified": 1,
        "unchanged": 1,
        "total": 4,
    }


def test_document_differ_includes_raw_unified_diff_lines() -> None:
    result = DocumentDiffer().compare(_version_a(), _version_b())
    modified = next(section for section in result.sections if section.status == "modified")

    assert any(line.startswith("---") for line in modified.diff_lines)
    assert any("-Old requirement" in line for line in modified.diff_lines)
    assert any("+New requirement" in line for line in modified.diff_lines)


def test_diff_summariser_uses_mocked_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = DocumentDiffer().compare(_version_a(), _version_b())
    summariser = DiffSummariser(provider="openai")

    async def fake_llm(prompt: str) -> str:
        if "Overall summary:" in prompt:
            return "Overall, the document changes one shared requirement."
        return "The shared section replaces the old requirement with a new one."

    monkeypatch.setattr(summariser, "_call_llm_with_retry", fake_llm)

    enriched = summariser.summarise(result)

    assert set(enriched.section_summaries) == {"Shared"}
    assert enriched.section_summaries["Shared"]
    assert enriched.overall_summary


def _version_a() -> list[Document]:
    return [
        _document("Shared", "Old requirement\nStill present", "v1"),
        _document("Removed", "This section was removed.", "v1"),
        _document("Stable", "No change here.", "v1"),
    ]


def _version_b() -> list[Document]:
    return [
        _document("Shared", "New requirement\nStill present", "v2"),
        _document("Stable", "No change here.", "v2"),
        _document("Added", "This section was added.", "v2"),
    ]


def _document(section_title: str, text: str, version_label: str) -> Document:
    return Document(
        text=text,
        metadata={
            "source_file": f"{version_label}.docx",
            "version_label": version_label,
            "page_number": 1,
            "section_title": section_title,
            "file_type": "docx",
        },
    )
