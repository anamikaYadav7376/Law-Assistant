"""
PDF Ingestion and Indexing Pipeline for DesignLaw AI.

This module processes the official 'Manual of Designs Practice & Procedure' PDF,
extracts text page-by-page while tracking verified chapter and section headings,
splits the content into structured chunks, and persists them into a ChromaDB
vector database using HuggingFace all-MiniLM-L6-v2 embeddings.
"""

import os
import re
import sys
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

DEFAULT_PDF_PATH = "law-Assistant.pdf"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "law_assistant_designs")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Known Chapter structure from Manual Table of Contents (Page 2) for verification
TOC_CHAPTERS = {
    1: "Chapter 1 – Introduction",
    2: "Chapter 2 – Important definitions",
    3: "Chapter 3 – Filing of Design Application",
    4: "Chapter 4 – Examination and registration of designs",
    5: "Chapter 5 – Patent Office Journal",
    6: "Chapter 6 – Piracy of registered designs and consequences thereof",
    7: "Chapter 7 – Post registration procedures",
    8: "Chapter 8 – Transfer of rights",
    9: "Chapter 9 – General Services",
    10: "Chapter 10 – Use and acquisition of registered design by Government",
    11: "Chapter 11 – General powers of Controller",
    12: "Chapter 12 – Evidences etc.",
    13: "Chapter 13 – Restrictive conditions in contract etc.",
    14: "Chapter 14 – Appeals",
    15: "Chapter 15 – Requirements before delivery on sales",
    16: "Chapter 16 – Timelines",
}

# Regex patterns for detecting chapters and numbered sections
CHAPTER_PATTERN = re.compile(r'^\s*CHAPTER\s*[-–—]?\s*(\d+)\s*$', re.IGNORECASE)
SECTION_PATTERN = re.compile(r'^\s*(\d{2}\.\d{2}(?:\.\d{2})?)\s+([A-Za-z0-9 ,/–—\(\)\-\.]+)')


def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Initialize and return the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def extract_pages_with_metadata(pdf_path: str) -> List[Tuple[int, str, Optional[str], Optional[str]]]:
    """
    Extract text page by page from the PDF and track verified chapter and section headings.
    
    Returns a list of tuples: (page_num, page_text, active_chapter, active_section)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open or parse PDF at {pdf_path}: {e}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"The PDF at {pdf_path} contains 0 pages.")

    pages_data = []
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    non_empty_pages = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed page number
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            print(f"[Warning] Error extracting text from page {page_num}: {e}")
            raw_text = ""

        clean_text = raw_text.strip()
        if clean_text:
            non_empty_pages += 1

        lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

        # 1. Check for Chapter heading on this page
        for line_idx, line in enumerate(lines[:10]):
            ch_match = CHAPTER_PATTERN.match(line)
            if ch_match:
                ch_num = int(ch_match.group(1))
                if ch_num in TOC_CHAPTERS:
                    current_chapter = TOC_CHAPTERS[ch_num]
                else:
                    # Look ahead for title
                    next_line = lines[line_idx + 1] if line_idx + 1 < len(lines) else ""
                    current_chapter = f"Chapter {ch_num} - {next_line}" if next_line else f"Chapter {ch_num}"
                # Reset section when entering a new chapter
                current_section = None
                break

        # Check for Annexure if applicable
        for line in lines[:5]:
            if line.startswith("Annexure") or "Important Case Laws" in line:
                if page_num >= 65:
                    current_chapter = "Annexure I – Important Case Laws"
                    current_section = None
            elif "Specimen novelty statements" in line:
                if page_num >= 86:
                    current_chapter = "Annexure II – Specimen novelty statements"
                    current_section = None

        # 2. Check for section heading on this page
        for line in lines:
            sec_match = SECTION_PATTERN.match(line)
            if sec_match:
                current_section = f"{sec_match.group(1)} {sec_match.group(2).strip()}"

        pages_data.append((page_num, clean_text, current_chapter, current_section))

    if non_empty_pages == 0:
        raise ValueError(f"Extracted text is empty or PDF is unreadable: {pdf_path}")

    return pages_data


def chunk_documents(
    pages_data: List[Tuple[int, str, Optional[str], Optional[str]]],
    source_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[Document], List[str]]:
    """
    Split extracted page text into chunks and preserve precise chapter, section,
    page number, and source metadata. Generates deterministic chunk IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    chunk_ids: List[str] = []

    for page_num, text, active_chapter, active_section in pages_data:
        if not text:
            continue

        raw_chunks = splitter.split_text(text)
        current_chunk_section = active_section

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            # Check if this specific chunk contains a section header
            for line in chunk_text.split("\n"):
                sec_match = SECTION_PATTERN.match(line.strip())
                if sec_match:
                    current_chunk_section = f"{sec_match.group(1)} {sec_match.group(2).strip()}"

            metadata: Dict[str, Any] = {
                "source": source_name,
                "page": page_num,
                "chapter": active_chapter if active_chapter else "N/A",
                "section": current_chunk_section if current_chunk_section else "N/A",
                "chunk_index": chunk_idx,
            }

            # Generate deterministic stable chunk ID to prevent duplicates
            content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:10]
            chunk_id = f"p{page_num}_c{chunk_idx}_{content_hash}"

            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)
            chunk_ids.append(chunk_id)

    return documents, chunk_ids


def ingest_pdf(
    pdf_path: str = DEFAULT_PDF_PATH,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[int, int, int]:
    """
    Run complete PDF ingestion pipeline:
    1. Reads PDF and extracts pages with chapter/section metadata.
    2. Splits into recursive chunks with stable IDs.
    3. Indexes chunks into persistent Chroma vector store.
    
    Returns (pages_loaded, chunks_created, chunks_indexed).
    """
    print(f"==================================================")
    print(f"Starting PDF Ingestion Pipeline")
    print(f"Source PDF         : {pdf_path}")
    print(f"Persistence Dir    : {persist_dir}")
    print(f"Chroma Collection  : {collection_name}")
    print(f"Embedding Model    : {EMBEDDING_MODEL_NAME}")
    print(f"==================================================")

    # 1. Extract pages
    pages_data = extract_pages_with_metadata(pdf_path)
    pages_loaded = len(pages_data)
    print(f"[1/3] Pages loaded successfully: {pages_loaded} pages")

    # 2. Chunk documents
    source_name = os.path.basename(pdf_path)
    documents, chunk_ids = chunk_documents(
        pages_data=pages_data,
        source_name=source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks_created = len(documents)
    print(f"[2/3] Chunks created: {chunks_created} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # 3. Embed and Index into Chroma
    print(f"[3/3] Initializing embeddings and vector storage...")
    embeddings = get_embedding_function()

    os.makedirs(persist_dir, exist_ok=True)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Ingest in batches to handle memory and provide progress
    batch_size = 64
    total_indexed = 0

    print(f"      Indexing {chunks_created} chunks into collection '{collection_name}'...")
    for i in range(0, chunks_created, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = chunk_ids[i : i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        total_indexed += len(batch_docs)
        print(f"      Progress: {total_indexed}/{chunks_created} chunks processed...")

    print(f"==================================================")
    print(f"Ingestion Completed Successfully!")
    print(f"  • Pages loaded   : {pages_loaded}")
    print(f"  • Chunks created : {chunks_created}")
    print(f"  • Chunks indexed : {total_indexed}")
    print(f"  • Vector store   : {persist_dir}")
    print(f"==================================================")

    return pages_loaded, chunks_created, total_indexed


if __name__ == "__main__":
    try:
        pdf_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
        ingest_pdf(pdf_path=pdf_file)
    except Exception as err:
        print(f"\n[FATAL ERROR] Ingestion failed: {err}", file=sys.stderr)
        sys.exit(1)
