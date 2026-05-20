# Document Compare RAG

Document Compare RAG ingests document versions, builds searchable vector indexes,
computes structured diffs, and produces plain-English summaries.

## Architecture

The app is organized into five layers:

1. Ingestion: `app/ingestion/` loads `.doc`, `.docx`, and `.pdf` files into
   normalized `Document` objects with source, version, page, section, and file
   type metadata.
2. RAG pipeline: `app/pipeline/` chunks documents, embeds chunks, and indexes
   each version in FAISS or Chroma.
3. Comparison engine: `app/comparison/` aligns sections, creates diffs with
   `difflib`, and enriches results with LLM summaries.
4. UI: `app/ui/streamlit_app.py` provides an interactive Streamlit workflow.
5. QA validation: `tests/` covers ingestion, chunking, embedding, vector store
   behavior, comparison, and mocked summarization.

Configuration is loaded from `.env` through the root `config.py` module.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
make install
```

Or directly:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with the values for your environment.

## External OCR Tools

Scanned PDF OCR requires system tools in addition to Python packages:

- Tesseract OCR must be installed and available to `pytesseract`.
- Poppler must be installed and available to `pdf2image`.

## Run The Streamlit UI

```bash
make run
```

Equivalent direct command:

```bash
streamlit run app/ui/streamlit_app.py
```

## Run The CLI

```bash
python -m app.main --file1 v1.pdf --label1 "v1" --file2 v2.docx --label2 "v2"
```

The CLI runs:

```text
ingest -> chunk -> embed -> index -> compare -> summarise
```

Use `--skip-summary` when you want to run the full local pipeline without an
LLM API call.

## Run Tests

```bash
make test
```

Equivalent direct command:

```bash
pytest tests/ -v
```

## Lint

```bash
make lint
```

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. Use `sentence-transformers/*` for local embeddings or `text-embedding-*` for OpenAI embeddings. |
| `LLM_PROVIDER` | No | `anthropic` | Summary provider. Supported values: `anthropic`, `openai`. |
| `VECTOR_STORE` | No | `faiss` | Vector store backend. Supported values: `faiss`, `chroma`. |
| `CHUNK_SIZE` | No | `1000` | Maximum chunk size in characters. |
| `CHUNK_OVERLAP` | No | `200` | Overlap between adjacent chunks in characters. |
| `OPENAI_API_KEY` | Conditional | empty | Required for OpenAI LLM summaries or OpenAI embeddings. |
| `ANTHROPIC_API_KEY` | Conditional | empty | Required when `LLM_PROVIDER=anthropic` and summaries are enabled. |

## Project Layout

```text
doc_compare_rag/
|-- app/
|   |-- comparison/
|   |-- ingestion/
|   |-- pipeline/
|   |-- ui/
|   `-- utils/
|-- tests/
|-- config.py
|-- requirements.txt
|-- Makefile
`-- README.md
```
