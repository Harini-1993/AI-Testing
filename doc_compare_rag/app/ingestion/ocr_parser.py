from pathlib import Path

from app.ingestion.document import Document


class OcrPdfParser:
    """Parse scanned PDFs by rendering pages and running OCR."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def parse(
        self,
        file_path: str | Path,
        *,
        source_file: str | None = None,
        version_label: str = "",
        file_type: str = "pdf_ocr",
    ) -> list[Document]:
        from pdf2image import convert_from_path
        import pytesseract

        path = Path(file_path)
        source = source_file or path.name

        try:
            images = convert_from_path(str(path), dpi=self.dpi)
        except Exception as exc:
            return [
                _make_document(
                    text="",
                    source_file=source,
                    version_label=version_label,
                    page_number=1,
                    section_title="",
                    file_type=file_type,
                    ocr_error=f"OCR failed while converting PDF pages: {exc}",
                )
            ]

        parsed: list[Document] = []

        for page_number, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image).strip()
                ocr_error = ""
            except Exception as exc:
                text = ""
                ocr_error = f"OCR failed on page {page_number}: {exc}"

            parsed.append(
                _make_document(
                    text=text,
                    source_file=source,
                    version_label=version_label,
                    page_number=page_number,
                    section_title="",
                    file_type=file_type,
                    ocr_error=ocr_error,
                )
            )

        return parsed


def parse_scanned_pdf(
    file_path: str | Path,
    *,
    source_file: str | None = None,
    version_label: str = "",
    file_type: str = "pdf_ocr",
    dpi: int = 300,
) -> list[Document]:
    return OcrPdfParser(dpi=dpi).parse(
        file_path,
        source_file=source_file,
        version_label=version_label,
        file_type=file_type,
    )


def _make_document(
    *,
    text: str,
    source_file: str,
    version_label: str,
    page_number: int,
    section_title: str,
    file_type: str,
    ocr_error: str = "",
) -> Document:
    metadata = {
        "source_file": source_file,
        "version_label": version_label,
        "page_number": page_number,
        "section_title": section_title,
        "file_type": file_type,
    }

    if ocr_error:
        metadata["ocr_error"] = ocr_error

    return Document(text=text, metadata=metadata)
