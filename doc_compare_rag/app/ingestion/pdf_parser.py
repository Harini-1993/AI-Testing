from pathlib import Path

from app.ingestion.document import Document


class TextPdfParser:
    """Parse text-extractable PDFs page by page."""

    def parse(
        self,
        file_path: str | Path,
        *,
        source_file: str | None = None,
        version_label: str = "",
        file_type: str = "pdf_text",
    ) -> list[Document]:
        import pdfplumber

        path = Path(file_path)
        source = source_file or path.name
        parsed: list[Document] = []
        current_section = ""

        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                stripped_text = text.strip()
                if not stripped_text:
                    continue

                detected = _detect_section_title(text)
                if detected:
                    current_section = detected

                parsed.append(
                    _make_document(
                        text=stripped_text,
                        source_file=source,
                        version_label=version_label,
                        page_number=page_number,
                        section_title=current_section,
                        file_type=file_type,
                    )
                )

        return parsed


def parse_pdf(
    file_path: str | Path,
    *,
    source_file: str | None = None,
    version_label: str = "",
    file_type: str = "pdf_text",
) -> list[Document]:
    return TextPdfParser().parse(
        file_path,
        source_file=source_file,
        version_label=version_label,
        file_type=file_type,
    )


def _detect_section_title(text: str) -> str:
    lines = text.splitlines()

    for index, line in enumerate(lines):
        candidate = line.strip()
        if not candidate:
            continue

        if _is_all_caps_heading(candidate):
            return candidate

        if index + 1 < len(lines) and not lines[index + 1].strip():
            return candidate

    return ""


def _is_all_caps_heading(line: str) -> bool:
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and line == line.upper()


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
