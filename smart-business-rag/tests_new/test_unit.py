from app.services.pdf_loader import extract_metadata_from_filename, clean_text

# ==========================================
# 2.1 PDF Loader & Sanitation Unit Tests
# ==========================================

def test_extract_metadata_standard_naming():
    """Verifies that standard filename parsing correctly extracts company, quarter, and filename."""
    meta = extract_metadata_from_filename("nubank-3q25.pdf")
    assert meta["company"] == "Nubank"
    assert meta["fiscal_period"] == "3Q25"
    assert meta["source_file"] == "nubank-3q25.pdf"


def test_extract_metadata_case_insensitivity():
    """Ensures uppercase/lowercase variations and extensions still resolve quarters properly."""
    meta = extract_metadata_from_filename("NuBank-4Q24.PDF")
    assert meta["company"] == "Nubank"
    assert meta["fiscal_period"] == "4Q24"


def test_extract_metadata_fallback_on_unknown():
    """Ensures that files without quarter patterns return 'Unknown' instead of raising an error."""
    meta = extract_metadata_from_filename("general_report.pdf")
    assert meta["company"] == "Nubank"
    assert meta["fiscal_period"] == "Unknown"


def test_clean_text_normalizes_whitespace():
    """Ensures erratic PDF spaces, tabs, and repeated blank newlines are collapsed."""
    raw = "Brazil   continues    to deliver.\n\n\n\nWith \t a customer base."
    expected = "Brazil continues to deliver.\n\nWith a customer base."
    assert clean_text(raw) == expected


def test_clean_text_empty_input():
    """Ensures empty strings or whitespace-only strings return cleanly as empty strings."""
    assert clean_text("") == ""
    assert clean_text("   \n\n   ") == ""

from langchain_core.documents import Document
from app.services.chunker import chunk_documents


# ==========================================
# 2.2 Chunker Mechanics Unit Tests
# ==========================================

def test_chunk_documents_splits_and_preserves_metadata():
    """Verifies that large documents split into multiple chunks, keep parent metadata, and get chunk_index."""
    long_content = (
        "Nubank active customers reached 110 million in Brazil with strong engagement. "
        "Credit portfolio expansion accelerated across credit cards and personal loans. "
    ) * 15  # ~2,000 characters

    parent_doc = Document(
        page_content=long_content,
        metadata={"source": "nubank-3q25.pdf", "page": 1, "fiscal_period": "3Q25", "company": "Nubank"},
    )

    chunks = chunk_documents([parent_doc], chunk_size=400, chunk_overlap=80)

    # Must produce multiple chunks
    assert len(chunks) > 1

    # Verify metadata inheritance and sequential chunk_index
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "nubank-3q25.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["fiscal_period"] == "3Q25"
        assert chunk.metadata["chunk_index"] == idx


def test_chunk_documents_single_chunk_boundary(sample_document: Document):
    """Verifies that a document smaller than chunk_size produces exactly one chunk with chunk_index = 0."""
    chunks = chunk_documents([sample_document], chunk_size=2000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source"] == sample_document.metadata["source"]
    assert chunks[0].page_content == sample_document.page_content


import pytest
from pydantic import ValidationError

from app.models.schemas import QueryRequest, Citation, OperationalInsight


# ==========================================
# 2.3 Pydantic Schemas & Contracts Unit Tests
# ==========================================

def test_query_request_valid():
    """Verifies that a valid query request payload is parsed and preserves all optional fields."""
    payload = {
        "query": "What is the net interest margin for 4Q25?",
        "fiscal_filter": "4Q25",
        "thread_id": "session-xyz-123",
    }
    req = QueryRequest(**payload)
    assert req.query == "What is the net interest margin for 4Q25?"
    assert req.fiscal_filter == "4Q25"
    assert req.thread_id == "session-xyz-123"


def test_query_request_min_length_validation():
    """Ensures queries with fewer than 3 characters fail Pydantic validation."""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="ab")
    assert "String should have at least 3 characters" in str(exc_info.value)


def test_query_request_missing_required_query():
    """Ensures omitting the required 'query' field raises a ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest(fiscal_filter="3Q25")  # type: ignore[call-arg]


def test_operational_insight_defaults():
    """Verifies default values for citations (empty list) and is_grounded (True)."""
    insight = OperationalInsight(answer="Nubank expanded credit lines.")
    assert insight.answer == "Nubank expanded credit lines."
    assert insight.citations == []
    assert insight.is_grounded is True


def test_operational_insight_nested_serialization(sample_insight: OperationalInsight):
    """Verifies nested Citation models and serialization via model_dump()."""
    dumped = sample_insight.model_dump()

    assert dumped["is_grounded"] is True
    assert len(dumped["citations"]) == 1
    assert dumped["citations"][0]["document"] == "nubank-3q25.pdf"
    assert dumped["citations"][0]["page"] == 1
    assert "110 million customers" in dumped["citations"][0]["quote"]

from app.services.agent_state import AgentState
from app.services.graph_builder import decide_to_generate


# ==========================================
# 2.4 LangGraph Isolated Router Guardrail
# ==========================================

def test_decide_to_generate_routes_to_generate(sample_document: Document):
    """Verifies that when relevant documents exist, the router proceeds to 'generate'."""
    state: AgentState = {
        "query": "What is the Brazil customer count?",
        "relevant_documents": [sample_document],
    }

    decision = decide_to_generate(state)
    assert decision == "generate"


def test_decide_to_generate_routes_to_fallback_on_empty():
    """Verifies that an empty relevant_documents list triggers the 'fallback' safety path."""
    state: AgentState = {
        "query": "What are the specs of a SpaceX Falcon Heavy?",
        "relevant_documents": [],
    }

    decision = decide_to_generate(state)
    assert decision == "fallback"


def test_decide_to_generate_routes_to_fallback_on_missing_key():
    """Verifies that if the 'relevant_documents' key is missing altogether, it safely falls back."""
    state: AgentState = {
        "query": "Tell me something unrelated.",
    }

    decision = decide_to_generate(state)
    assert decision == "fallback"