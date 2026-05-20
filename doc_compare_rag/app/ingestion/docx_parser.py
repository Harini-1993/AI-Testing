from pathlib import Path
from typing import Any

from app.ingestion.document import Document


class DocxParser:
    """Parse .docx files into normalized ingestion documents."""

    HEADING_STYLES = {"Heading 1", "Heading 2"}

    def parse(
        self,
        file_path: str | Path,
        *,
        source_file: str | None = None,
        version_label: str = "",
        file_type: str = "docx",
    ) -> list[Document]:
        from docx import Document as PythonDocxDocument
        from docx.table import Table

        path = Path(file_path)
        source = source_file or path.name
        document = PythonDocxDocument(str(path))
        parsed: list[Document] = []
        current_section = ""

        for block in _iter_block_items(document):
            if _is_paragraph(block):
                text = block.text.strip()
                if not text:
                    continue

                if block.style and block.style.name in self.HEADING_STYLES:
                    current_section = text

                parsed.append(
                    _make_document(
                        text=text,
                        source_file=source,
                        version_label=version_label,
                        page_number=1,
                        section_title=current_section,
                        file_type=file_type,
                    )
                )
                continue

            if isinstance(block, Table):
                table_text = _extract_table_text(block)
                if table_text:
                    parsed.append(
                        _make_document(
                            text=table_text,
                            source_file=source,
                            version_label=version_label,
                            page_number=1,
                            section_title=current_section,
                            file_type=file_type,
                        )
                    )

        return parsed


def parse_docx(
    file_path: str | Path,
    *,
    source_file: str | None = None,
    version_label: str = "",
    file_type: str = "docx",
) -> list[Document]:
    return DocxParser().parse(
        file_path,
        source_file=source_file,
        version_label=version_label,
        file_type=file_type,
    )


def _iter_block_items(document: Any):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _is_paragraph(block: Any) -> bool:
    return block.__class__.__name__ == "Paragraph"


def _extract_table_text(table: Any) -> str:
    rows: list[str] = []

    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def _make_document(
    *,
    text: str,
    source_file: str,
    version_label: str,
    page_number: int,
    section_title: str,
    file_type: str,
) -> Document:
    return Document(
        text=text,
        metadata={
            "source_file": source_file,
            "version_label": version_label,
            "page_number": page_number,
            "section_title": section_title,
            "file_type": file_type,
        },
    )
