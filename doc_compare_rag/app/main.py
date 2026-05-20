"""Main CLI entry point for document comparison and RAG analysis.

Provides a command-line interface for running the complete pipeline:
ingestion -> chunking -> embedding -> indexing -> comparison -> summarization.

Usage:
    python -m app.main --file1 v1.pdf --label1 "v1.0" \
                       --file2 v2.docx --label2 "v2.0"
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from app.comparison.differ import DocumentDiffer
from app.comparison.summariser import DiffSummariser, EnrichedComparisonResult
from app.ingestion.document import Document
from app.ingestion.loader import DocumentLoader
from app.pipeline.chunker import Chunk, TextChunker
from app.pipeline.embedder import Embedder
from app.pipeline.vector_store import VectorStoreManager
from app.utils.helpers import truncate_text


logger = logging.getLogger(__name__)


class DocumentComparator:
    """Pipeline orchestrator for document comparison and RAG indexing."""

    def __init__(
        self,
        vector_backend: str = "faiss",
        skip_summary: bool = False,
        config: Config | None = None,
    ) -> None:
        """Initialize pipeline components."""
        self.config = config or Config.from_env(validate=False)
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.vector_store = VectorStoreManager(backend=vector_backend)
        self.differ = DocumentDiffer()
        self.summariser = DiffSummariser()
        self.skip_summary = skip_summary

    def run(
        self,
        file1_path: str,
        label1: str,
        file2_path: str,
        label2: str,
    ) -> None:
        """Run the complete comparison pipeline and print a formatted report."""
        try:
            print("\n" + "=" * 70)
            print("DOCUMENT COMPARISON REPORT")
            print("=" * 70)

            print("\n[1/6] Loading documents...")
            docs_v1 = self.loader.load(file1_path, version_label=label1)
            docs_v2 = self.loader.load(file2_path, version_label=label2)
            self._print_loaded(file1_path, docs_v1)
            self._print_loaded(file2_path, docs_v2)

            print("\n[2/6] Chunking documents...")
            chunks_v1 = self.chunker.split(docs_v1)
            chunks_v2 = self.chunker.split(docs_v2)
            print(f"  [ok] Created {len(chunks_v1)} chunks from {label1}")
            print(f"  [ok] Created {len(chunks_v2)} chunks from {label2}")

            print("\n[3/6] Generating embeddings...")
            embeddings_v1 = self.embedder.embed(chunks_v1)
            embeddings_v2 = self.embedder.embed(chunks_v2)
            print(f"  [ok] Generated {len(embeddings_v1)} embeddings for {label1}")
            print(f"  [ok] Generated {len(embeddings_v2)} embeddings for {label2}")

            print("\n[4/6] Indexing vectors...")
            self._index_version(label1, chunks_v1, embeddings_v1)
            self._index_version(label2, chunks_v2, embeddings_v2)
            print(f"  [ok] Indexed {len(chunks_v1)} vectors for {label1}")
            print(f"  [ok] Indexed {len(chunks_v2)} vectors for {label2}")

            print("\n[5/6] Comparing documents...")
            comparison_result = self.differ.compare(docs_v1, docs_v2)
            self._print_comparison_stats(comparison_result.summary_stats)

            print("\n[6/6] Generating AI summaries...")
            if self.skip_summary:
                enriched_result = EnrichedComparisonResult(
                    version_a_label=comparison_result.version_a_label,
                    version_b_label=comparison_result.version_b_label,
                    sections=comparison_result.sections,
                    summary_stats=comparison_result.summary_stats,
                    section_summaries={},
                    overall_summary="Summary generation skipped.",
                )
                print("  [ok] Summary generation skipped")
            else:
                enriched_result = self.summariser.summarise(comparison_result)
                print(
                    "  [ok] Generated "
                    f"{len(enriched_result.section_summaries)} section summaries"
                )
                print("  [ok] Generated overall summary")

            self._print_report(enriched_result, label1, label2)

        except FileNotFoundError as exc:
            print(f"Error: File not found - {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"Error: Invalid input - {exc}", file=sys.stderr)
            sys.exit(2)
        except Exception as exc:
            logger.exception("Unexpected error during comparison")
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(3)

    def _print_loaded(self, file_path: str, documents: list[Document]) -> None:
        """Print a concise ingestion status line."""
        print(f"  [ok] Loaded {len(documents)} document records from {Path(file_path).name}")

    def _index_version(
        self,
        label: str,
        chunks: list[Chunk],
        embeddings: list[object],
    ) -> None:
        """Index a version only when chunking produced searchable content."""
        if not chunks:
            logger.warning("No chunks produced for %s; skipping vector index", label)
            return

        self.vector_store.index(label, chunks, embeddings)

    def _print_comparison_stats(self, stats: dict[str, int]) -> None:
        """Print comparison counters."""
        print("  [ok] Comparison complete:")
        print(f"    - Total sections: {stats.get('total', 0)}")
        print(f"    - Modified: {stats.get('modified', 0)}")
        print(f"    - Added: {stats.get('added', 0)}")
        print(f"    - Removed: {stats.get('removed', 0)}")
        print(f"    - Unchanged: {stats.get('unchanged', 0)}")

    def _print_report(
        self,
        enriched_result: EnrichedComparisonResult,
        label1: str,
        label2: str,
    ) -> None:
        """Print a formatted comparison report to stdout."""
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        print(f"\nComparing: {label1} -> {label2}")
        print("\nSummary Statistics:")
        print(f"  - Total sections: {enriched_result.summary_stats['total']}")
        print(f"  - Modified: {enriched_result.summary_stats['modified']}")
        print(f"  - Added: {enriched_result.summary_stats['added']}")
        print(f"  - Removed: {enriched_result.summary_stats['removed']}")
        print(f"  - Unchanged: {enriched_result.summary_stats['unchanged']}")

        print("\nOverall Summary:")
        print(f"  {enriched_result.overall_summary}")

        print("\nSection-by-Section Changes:")
        print("-" * 70)

        for index, section in enumerate(enriched_result.sections, start=1):
            section_title = section.section_title or f"Section {index}"
            print(f"\n{index}. [{section.status.upper()}] {section_title}")

            summary = enriched_result.section_summaries.get(section.section_title)
            if summary:
                print(f"   Summary: {truncate_text(summary, max_chars=150)}")

            if section.status == "modified":
                words_a = len(section.text_a.split()) if section.text_a else 0
                words_b = len(section.text_b.split()) if section.text_b else 0
                word_delta = words_b - words_a
                delta_str = f"+{word_delta}" if word_delta >= 0 else str(word_delta)
                print(f"   Word count: {words_a} -> {words_b} ({delta_str})")

        print("\n" + "=" * 70)
        print("END OF REPORT")
        print("=" * 70 + "\n")


def main() -> None:
    """Parse CLI arguments and run the comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Compare two documents and generate a detailed analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main --file1 report_v1.pdf --label1 "v1.0" \
                     --file2 report_v2.pdf --label2 "v2.0"

  python -m app.main --file1 doc.docx --label1 "draft" \
                     --file2 doc_final.docx --label2 "final"
        """,
    )
    parser.add_argument(
        "--file1",
        required=True,
        help="Path to first document (PDF, DOCX, or DOC)",
    )
    parser.add_argument(
        "--label1",
        default="v1.0",
        help="Version label for first document (default: v1.0)",
    )
    parser.add_argument(
        "--file2",
        required=True,
        help="Path to second document (PDF, DOCX, or DOC)",
    )
    parser.add_argument(
        "--label2",
        default="v2.0",
        help="Version label for second document (default: v2.0)",
    )
    parser.add_argument(
        "--vector-store",
        choices=["faiss", "chroma"],
        default="faiss",
        help="Vector store backend to use (default: faiss)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip LLM summarization after comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = Config.from_env(validate=False)
    config.validate(require_llm_key=not args.skip_summary)

    comparator = DocumentComparator(
        vector_backend=args.vector_store,
        skip_summary=args.skip_summary,
        config=config,
    )
    comparator.run(args.file1, args.label1, args.file2, args.label2)


if __name__ == "__main__":
    main()
