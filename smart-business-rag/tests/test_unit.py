import pytest
from pydantic import ValidationError
from langchain_core.documents import Document

from app.services.pdf_loader import extract_metadata_from_filename, clean_text
from app.services.chunker import chunk_documents
from app.models.schemas import QueryRequest, Citation, OperationalInsight


# ==========================================
# 1. PDF Loader & Metadata Unit Tests
# ==========================================
def test_extract_metadata_standard_naming():
    """Verifies that filename parsing correctly identifies company and quarter."""
    meta = extract_metadata_from_filename("nubank-3q25.pdf")
    assert meta["company"] == "Nubank"
    assert meta["fiscal_period"] == "3Q25"
    assert meta["source_file"] == "nubank-3q25.pdf"

def test_extract_metadata_case_insensitivity():
    """Ensures uppercase/lowercase variations still resolve quarters properly."""
    meta = extract_metadata_from_filename("NuBank-4Q24.PDF")
    assert meta["company"] == "Nubank"
    assert meta["fiscal_period"] == "4Q24"

def test_clean_text_normalizes_whitespace():
    """Ensures erratic PDF spaces and repeated newlines are collapsed."""
    raw = "Brazil   continues    to deliver.\n\n\n\nWith a customer base."
    expected = "Brazil continues to deliver.\n\nWith a customer base."
    assert clean_text(raw) == expected


# ==========================================
# 2. Text Chunker Unit Tests
# ==========================================
def test_chunk_documents_preserves_metadata():
    """Verifies that chunks inherit page metadata and gain unique chunk indices."""
    dummy_doc = Document(
        page_content="Nubank active customers reached 110 million. " * 30,  # ~1350 characters
        metadata={"source": "test.pdf", "page": 1, "fiscal_period": "3Q25", "company": "Nubank"}
    )
    
    chunks = chunk_documents([dummy_doc], chunk_size=500, chunk_overlap=100)
    
    # Must split into multiple chunks
    assert len(chunks) > 1
    
    # Each chunk must preserve the parent metadata and add chunk_index
    for idx, c in enumerate(chunks):
        assert c.metadata["source"] == "test.pdf"
        assert c.metadata["page"] == 1
        assert c.metadata["fiscal_period"] == "3Q25"
        assert c.metadata["chunk_index"] == idx


# ==========================================
# 3. Pydantic Schema Validation Tests
# ==========================================
def test_query_request_valid():
    """Valid payload should parse cleanly without errors."""
    req = QueryRequest(query="What is the net income?", fiscal_filter="3Q25")
    assert req.query == "What is the net income?"
    assert req.fiscal_filter == "3Q25"

def test_query_request_validation_failure():
    """Queries under 3 characters must be rejected by Pydantic."""
    with pytest.raises(ValidationError):
        QueryRequest(query="hi")  # Violates min_length=3

def test_operational_insight_schema():
    """Verifies nested Citation models inside OperationalInsight."""
    citation = Citation(document="test.pdf", page=2, quote="Growth was 15%")
    insight = OperationalInsight(
        answer="The growth achieved was strong.",
        citations=[citation],
        is_grounded=True
    )
    assert insight.is_grounded is True
    assert len(insight.citations) == 1
    assert insight.citations[0].page == 2