"""
Unit and integration tests for RAG retrieval, Gemini grounding, citations, and Not found behavior.
"""

import os
import pytest
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from rag_chain import (
    retrieve_relevant_chunks,
    check_evidence_sufficiency,
    verify_citation_against_evidence,
    answer_legal_question,
    get_gemini_client,
    check_database_ready,
)


class MockSupportedLLM:
    """Mock LLM returning a valid, grounded answer format."""
    def invoke(self, messages):
        return AIMessage(content="""Answer: The initial copyright in a registered design lasts for ten years from the registration date.

Reference: Chapter 6 – Piracy of registered designs and consequences thereof, Section 06.02 Copyright in design (rights of registered proprietor), PDF page 35.

Supporting excerpt: "The proprietor of the design gets exclusive right to apply the design to the article in a class in which the design is registered." """)


class MockNotFoundLLM:
    """Mock LLM returning Not found when unsupported."""
    def invoke(self, messages):
        return AIMessage(content="Not found")


def test_retrieval_of_relevant_passages():
    """Test 3: Retrieval returns top relevant chunks with valid metadata and scores."""
    is_ready, _ = check_database_ready()
    assert is_ready, "Vector database must be indexed for retrieval test."

    query = "What is the period of copyright in a registered design?"
    docs, scores = retrieve_relevant_chunks(query, top_k=5)

    assert len(docs) == 5
    assert len(scores) == 5

    # Check score and metadata structure
    for doc in docs:
        assert "page" in doc.metadata
        assert "chapter" in doc.metadata
        assert "section" in doc.metadata
        assert len(doc.page_content) > 0


def test_supported_question_with_valid_reference():
    """Test 4: Supported question returns structured answer with accurate citations."""
    query = "What is the initial period of copyright in a registered design?"
    res = answer_legal_question(query, llm_instance=MockSupportedLLM())

    assert res["is_supported"] is True
    assert res["answer"] != "Not found"
    assert "PDF page" in res["reference"]
    assert len(res["excerpt"]) > 0
    assert "Answer:" in res["full_text"]
    assert "Source:" in res["full_text"] or "Reference:" in res["full_text"]


def test_unsupported_question_returns_exactly_not_found():
    """Test 5: An unsupported question returns exactly 'Not found'."""
    # Completely off-topic question fails evidence sufficiency check
    query = "What is the recipe for cooking spaghetti carbonara?"
    res = answer_legal_question(query, llm_instance=MockNotFoundLLM())

    assert res["is_supported"] is False
    assert res["full_text"] == "Not found"
    assert res["answer"] == "Not found"


def test_passage_mentions_topic_but_does_not_answer():
    """Test 6: When retrieved passages mention terms but do not answer, returns Not found."""
    # Query passes sufficiency check due to keyword overlap, but LLM identifies absence of answer
    query = "Does the manual specify the names of all Controller Generals from 1950 to 1980?"
    res = answer_legal_question(query, llm_instance=MockNotFoundLLM())

    assert res["is_supported"] is False
    assert res["full_text"] == "Not found"
    assert res["answer"] == "Not found"


def test_missing_or_invalid_api_configuration():
    """Test 7: Missing or invalid Gemini API key raises clear, helpful error without leaking key."""
    # Temporarily unset key
    old_key = os.environ.get("GEMINI_API_KEY")
    try:
        os.environ["GEMINI_API_KEY"] = ""
        with pytest.raises(ValueError) as excinfo:
            get_gemini_client(api_key="")
        assert "Gemini API configuration is missing" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo_placeholder:
            get_gemini_client(api_key="your_gemini_api_key_here")
        assert "Gemini API configuration is missing" in str(excinfo_placeholder.value)
    finally:
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key


def test_citation_accuracy():
    """Test 9: Verifies that citations match retrieved document metadata."""
    sample_docs = [
        Document(
            page_content="The registration of a design shall be for a period of ten years.",
            metadata={
                "page": 35,
                "chapter": "Chapter 6 – Piracy of registered designs",
                "section": "06.02 Copyright in design",
                "source": "law-Assistant.pdf"
            }
        )
    ]

    raw_response = (
        "Answer: Registration lasts for 10 years.\n\n"
        "Reference: Chapter 6 – Piracy of registered designs, Section 06.02 Copyright in design, PDF page 35.\n\n"
        "Supporting excerpt: \"The registration of a design shall be for a period of ten years.\""
    )

    verified = verify_citation_against_evidence(raw_response, sample_docs)
    assert verified["is_supported"] is True
    assert "PDF page 35" in verified["reference"]
    assert "Chapter 6" in verified["reference"]
    assert verified["excerpt"] == "The registration of a design shall be for a period of ten years."
