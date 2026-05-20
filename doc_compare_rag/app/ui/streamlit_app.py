from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from app.comparison.differ import diff_text
from app.comparison.summariser import summarise_diff
from app.ingestion.document import Document
from app.ingestion.loader import load_document


def save_upload(uploaded_file: object) -> Path:
    """Persist a Streamlit upload to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def document_text(documents: list[Document]) -> str:
    return "\n\n".join(document.text for document in documents if document.text.strip())


def main() -> None:
    """Run the Streamlit document comparison app."""
    st.set_page_config(page_title="Document Compare RAG", layout="wide")
    st.title("Document Compare RAG")

    original_upload = st.file_uploader(
        "Original document",
        type=["doc", "docx", "pdf"],
    )
    revised_upload = st.file_uploader(
        "Revised document",
        type=["doc", "docx", "pdf"],
    )
    use_ocr = st.checkbox("Use OCR for PDFs")

    if st.button("Compare") and original_upload and revised_upload:
        original_path = save_upload(original_upload)
        revised_path = save_upload(revised_upload)

        original_text = document_text(load_document(original_path, use_ocr=use_ocr))
        revised_text = document_text(load_document(revised_path, use_ocr=use_ocr))
        diff = diff_text(original_text, revised_text)

        st.subheader("Structural Diff")
        st.code(diff or "No differences found.", language="diff")

        if diff and st.button("Summarise Diff"):
            st.subheader("Plain-Language Summary")
            st.write(summarise_diff(diff))


if __name__ == "__main__":
    main()
