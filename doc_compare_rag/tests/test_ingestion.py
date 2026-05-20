import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.ingestion.docx_parser import DocxParser
from app.ingestion.document import Document
from app.ingestion.loader import DocumentLoader, load_document
from app.ingestion.ocr_parser import OcrPdfParser
from app.ingestion.pdf_parser import TextPdfParser


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_heading("Introduction", level=1)
    document.add_paragraph("This is the first paragraph.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Cell A"
    table.cell(0, 1).text = "Cell B"
    document.save(path)
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    fpdf = pytest.importorskip("fpdf")
    pytest.importorskip("pdfplumber")

    path = tmp_path / "sample.pdf"
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "INTRODUCTION\n\nThis PDF contains extractable text.")
    pdf.output(str(path))
    return path


def test_docx_parser_extracts_paragraphs_headings_and_tables(docx_file: Path) -> None:
    documents = DocxParser().parse(docx_file, version_label="v1")

    assert [document.text for document in documents] == [
        "Introduction",
        "This is the first paragraph.",
        "Cell A | Cell B",
    ]
    assert all(document.metadata["section_title"] == "Introduction" for document in documents)
    assert all_metadata_fields_populated(documents)


def test_pdf_parser_extracts_pages_and_detects_section_titles(pdf_file: Path) -> None:
    documents = TextPdfParser().parse(pdf_file, version_label="v1")

    assert len(documents) == 1
    assert "extractable text" in documents[0].text
    assert documents[0].metadata["section_title"] == "INTRODUCTION"
    assert all_metadata_fields_populated(documents)


def test_ocr_pdf_parser_uses_pytesseract_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    image_to_string = MagicMock(return_value="OCR text")

    pdf2image = ModuleType("pdf2image")
    pdf2image.convert_from_path = MagicMock(return_value=["page-image"])
    pytesseract = ModuleType("pytesseract")
    pytesseract.image_to_string = image_to_string
    monkeypatch.setitem(sys.modules, "pdf2image", pdf2image)
    monkeypatch.setitem(sys.modules, "pytesseract", pytesseract)

    documents = OcrPdfParser().parse(pdf_path, version_label="v2")

    image_to_string.assert_called_once_with("page-image")
    assert documents[0].text == "OCR text"
    assert documents[0].metadata["file_type"] == "pdf_ocr"
    assert all_metadata_fields_populated(documents)


def test_document_loader_routes_docx(tmp_path: Path) -> None:
    parser = RecordingParser()
    path = tmp_path / "sample.docx"
    path.write_text("placeholder", encoding="utf-8")

    documents = DocumentLoader(docx_parser=parser).load(path, version_label="v1")

    assert parser.calls == [(path, "sample.docx", "v1", "docx")]
    assert documents[0].metadata["file_type"] == "docx"


def test_document_loader_routes_text_pdf(tmp_path: Path) -> None:
    text_parser = RecordingParser()
    ocr_parser = RecordingParser()
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")
    loader = DocumentLoader(text_pdf_parser=text_parser, ocr_pdf_parser=ocr_parser)
    loader._is_text_extractable_pdf = lambda _: True

    documents = loader.load(path, version_label="v1")

    assert text_parser.calls == [(path, "sample.pdf", "v1", "pdf_text")]
    assert ocr_parser.calls == []
    assert documents[0].metadata["file_type"] == "pdf_text"


def test_document_loader_routes_scanned_pdf_to_ocr(tmp_path: Path) -> None:
    text_parser = RecordingParser()
    ocr_parser = RecordingParser()
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4")
    loader = DocumentLoader(text_pdf_parser=text_parser, ocr_pdf_parser=ocr_parser)
    loader._is_text_extractable_pdf = lambda _: False

    documents = loader.load(path, version_label="v2")

    assert text_parser.calls == []
    assert ocr_parser.calls == [(path, "scan.pdf", "v2", "pdf_ocr")]
    assert documents[0].metadata["file_type"] == "pdf_ocr"


def test_document_loader_rejects_unsupported_format(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(sample)


def test_metadata_fields_are_always_populated() -> None:
    documents = [
        Document(
            text="hello",
            metadata={
                "source_file": "sample.docx",
                "version_label": "v1",
                "page_number": 1,
                "section_title": "",
                "file_type": "docx",
            },
        )
    ]

    assert all_metadata_fields_populated(documents)


class RecordingParser:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, str, str]] = []

    def parse(
        self,
        file_path: str | Path,
        *,
        source_file: str | None = None,
        version_label: str = "",
        file_type: str = "",
    ) -> list[Document]:
        path = Path(file_path)
        source = source_file or path.name
        self.calls.append((path, source, version_label, file_type))
        return [
            Document(
                text="parsed",
                metadata={
                    "source_file": source,
                    "version_label": version_label,
                    "page_number": 1,
                    "section_title": "",
                    "file_type": file_type,
                },
            )
        ]


def all_metadata_fields_populated(documents: list[Document]) -> bool:
    required = {
        "source_file",
        "version_label",
        "page_number",
        "section_title",
        "file_type",
    }
    return all(required <= set(document.metadata) for document in documents)
