import argparse
import json
import tempfile
from pathlib import Path

from app.ingestion.document import Document
from app.ingestion.docx_parser import DocxParser
from app.ingestion.ocr_parser import OcrPdfParser
from app.ingestion.pdf_parser import TextPdfParser


class DocumentLoader:
    """Detect document formats and route files to the appropriate parser."""

    def __init__(
        self,
        *,
        docx_parser: DocxParser | None = None,
        text_pdf_parser: TextPdfParser | None = None,
        ocr_pdf_parser: OcrPdfParser | None = None,
    ) -> None:
        self.docx_parser = docx_parser or DocxParser()
        self.text_pdf_parser = text_pdf_parser or TextPdfParser()
        self.ocr_pdf_parser = ocr_pdf_parser or OcrPdfParser()

    def load(self, file_path: str | Path, version_label: str = "") -> list[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        source_file = path.name

        if suffix == ".docx":
            return self.docx_parser.parse(
                path,
                source_file=source_file,
                version_label=version_label,
                file_type="docx",
            )

        if suffix == ".doc":
            with tempfile.TemporaryDirectory() as temp_dir:
                converted_path = Path(temp_dir) / f"{path.stem}.docx"
                _convert_doc_to_docx(path, converted_path)
                return self.docx_parser.parse(
                    converted_path,
                    source_file=source_file,
                    version_label=version_label,
                    file_type="docx",
                )

        if suffix == ".pdf":
            if self._is_text_extractable_pdf(path):
                return self.text_pdf_parser.parse(
                    path,
                    source_file=source_file,
                    version_label=version_label,
                    file_type="pdf_text",
                )

            return self.ocr_pdf_parser.parse(
                path,
                source_file=source_file,
                version_label=version_label,
                file_type="pdf_ocr",
            )

        raise ValueError(f"Unsupported file format: {suffix or '<none>'}")

    def _is_text_extractable_pdf(self, path: Path) -> bool:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return False

            first_page_text = pdf.pages[0].extract_text() or ""
            return len(first_page_text.strip()) > 50


def load_document(
    file_path: str | Path,
    use_ocr: bool | None = None,
    version_label: str = "",
) -> list[Document]:
    """Compatibility wrapper around DocumentLoader.

    The use_ocr argument is accepted for older callers; format routing is now
    automatic based on extension and PDF text extraction.
    """
    return DocumentLoader().load(file_path, version_label=version_label)


def _convert_doc_to_docx(source_path: Path, target_path: Path) -> None:
    import mammoth
    from docx import Document as PythonDocxDocument

    try:
        with source_path.open("rb") as document_file:
            result = mammoth.extract_raw_text(document_file)
    except Exception as exc:
        raise ValueError(f"Could not convert .doc file with mammoth: {exc}") from exc

    docx_document = PythonDocxDocument()
    for paragraph in result.value.splitlines():
        if paragraph.strip():
            docx_document.add_paragraph(paragraph.strip())

    docx_document.save(str(target_path))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Load a document for ingestion.")
    parser.add_argument("file_path", help="Path to a .doc, .docx, or .pdf file")
    parser.add_argument(
        "--version",
        dest="version_label",
        default="",
        help='Version label to attach to metadata, for example "v1"',
    )
    args = parser.parse_args()

    documents = DocumentLoader().load(args.file_path, version_label=args.version_label)
    print(
        json.dumps(
            [
                {
                    "text": document.text,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
