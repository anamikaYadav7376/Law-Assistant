"""
Unit and integration tests for PDF ingestion and metadata preservation.
"""

import os
import shutil
import pytest
from ingest import (
    extract_pages_with_metadata,
    chunk_documents,
    ingest_pdf,
    DEFAULT_PDF_PATH,
    TOC_CHAPTERS,
)


def test_pdf_loading_and_page_metadata():
    """Test 1: Confirm PDF can be loaded and page numbers are preserved accurately."""
    assert os.path.exists(DEFAULT_PDF_PATH), f"Expected PDF at {DEFAULT_PDF_PATH}"
    pages_data = extract_pages_with_metadata(DEFAULT_PDF_PATH)

    assert len(pages_data) == 87, f"Expected 87 pages in manual, got {len(pages_data)}"

    # Check 1-indexing of pages
    page_numbers = [p[0] for p in pages_data]
    assert page_numbers[0] == 1
    assert page_numbers[-1] == 87

    # Check chapter tracking: page 5 is Chapter 1
    page_5 = pages_data[4]
    assert page_5[0] == 5
    assert "Chapter 1" in (page_5[2] or "")


def test_chunking_and_preservation_of_metadata():
    """Test 2: Verify chunking preserves page, chapter, section, and source metadata."""
    pages_data = extract_pages_with_metadata(DEFAULT_PDF_PATH)
    sample_pages = pages_data[6:10]  # Pages 7 to 10 (Chapter 2)

    chunks, chunk_ids = chunk_documents(
        pages_data=sample_pages,
        source_name="law-Assistant.pdf",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) > 0
    assert len(chunks) == len(chunk_ids)

    # Validate metadata fields on every chunk
    for doc in chunks:
        meta = doc.metadata
        assert "source" in meta and meta["source"] == "law-Assistant.pdf"
        assert "page" in meta and 7 <= meta["page"] <= 10
        assert "chapter" in meta and "Chapter 2" in meta["chapter"]
        assert "section" in meta
        assert "chunk_index" in meta


def test_missing_pdf_and_empty_text_errors(tmp_path):
    """Test 8: Ensure missing PDF and invalid empty files raise clear errors."""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        extract_pages_with_metadata("non_existent_file.pdf")

    # Empty / corrupt file
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"Not a real PDF file content")

    with pytest.raises(ValueError):
        extract_pages_with_metadata(str(corrupt_pdf))


def test_duplicate_free_repeated_ingestion(tmp_path):
    """Test 10: Indexing the same document repeatedly produces no duplicate chunks in Chroma."""
    test_chroma_dir = str(tmp_path / "test_chroma_dedup")
    test_collection = "test_dedup_col"

    # Ingest once
    pages_1, chunks_1, indexed_1 = ingest_pdf(
        pdf_path=DEFAULT_PDF_PATH,
        persist_dir=test_chroma_dir,
        collection_name=test_collection,
    )
    assert indexed_1 > 0

    # Ingest a second time into the same collection
    pages_2, chunks_2, indexed_2 = ingest_pdf(
        pdf_path=DEFAULT_PDF_PATH,
        persist_dir=test_chroma_dir,
        collection_name=test_collection,
    )

    # Re-open Chroma and verify total count did not double
    from langchain_chroma import Chroma
    from ingest import get_embedding_function

    store = Chroma(
        collection_name=test_collection,
        embedding_function=get_embedding_function(),
        persist_directory=test_chroma_dir,
    )

    final_count = store._collection.count()
    assert final_count == chunks_1, f"Expected {chunks_1} chunks, but found {final_count} (duplicates were inserted)"

    # Cleanup
    shutil.rmtree(test_chroma_dir, ignore_errors=True)
