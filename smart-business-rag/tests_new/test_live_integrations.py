import os
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# ==========================================
# 4.1 Live Model Connectivity Tests
# ==========================================

def test_groq_connectivity():
    """Pings Groq API with gpt-oss-120b to verify API key and service availability."""
    groq = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
    )
    response = groq.invoke("Respond with exactly: 'Groq is operational.'")

    content = response.content if isinstance(response.content, str) else str(response.content)
    assert len(content.strip()) > 0
    assert "operational" in content.lower()


def test_gemini_connectivity():
    """Pings Google Gemini to verify API key and service availability."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("BACKUP_GOOGLE_API_KEY")
    assert api_key, "Neither GOOGLE_API_KEY nor BACKUP_GOOGLE_API_KEY is set in environment."

    gemini = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.0,
    )
    response = gemini.invoke("Respond with exactly: 'Gemini is operational.'")

    # Safely extract text whether response.content is str or list
    if isinstance(response.content, list):
        content = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in response.content)
    else:
        content = str(response.content)

    assert len(content.strip()) > 0
    assert "operational" in content.lower()

from sqlalchemy import create_engine, text


# ==========================================
# 4.2 Live Supabase & pgvector Tests
# ==========================================

def test_database_connection():
    """Verifies live connectivity to Supabase PostgreSQL and checks version."""
    database_url = os.getenv("DATABASE_URL")
    assert database_url, "DATABASE_URL is not set in .env"

    engine = create_engine(database_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).fetchone()
        assert res is not None, "Failed to read database version."
        version_str = str(res[0])
        assert "postgresql" in version_str.lower(), f"Unexpected database type: {version_str}"


def test_pgvector_extension_active():
    """Verifies that the pgvector extension is enabled and ready to store embeddings."""
    database_url = os.getenv("DATABASE_URL")
    assert database_url, "DATABASE_URL is not set in .env"

    engine = create_engine(database_url)
    with engine.connect() as conn:
        vector_check = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        ).fetchone()

        assert vector_check is not None, (
            "pgvector extension is NOT active in your Supabase DB! "
            "Run 'CREATE EXTENSION IF NOT EXISTS vector;' in Supabase SQL editor."
        )
        assert vector_check[0] == "vector"


from pathlib import Path
from app.services.pdf_loader import load_pdf_with_metadata
from app.services.chunker import chunk_documents


# ==========================================
# 4.3 Real Document Ingestion Tests
# ==========================================

def test_real_pdf_extraction_and_chunking(data_dir: Path):
    """Verifies that the actual corporate filings in data/ are parseable and chunkable."""
    pdf_files = list(data_dir.glob("*.pdf"))
    assert len(pdf_files) >= 1, f"No PDF files found in {data_dir}!"

    all_pages = []
    for pdf_path in pdf_files:
        pages = load_pdf_with_metadata(str(pdf_path))
        assert len(pages) > 0, f"Failed to extract any pages from {pdf_path.name}"
        all_pages.extend(pages)

    # Verify extracted page metadata
    for page in all_pages:
        assert page.metadata["company"] == "Nubank"
        assert page.metadata["fiscal_period"] in ["2Q25", "3Q25", "4Q25", "Unknown"]
        assert page.metadata["page"] >= 1
        assert len(page.page_content.strip()) > 0

    # Test chunking against the real extracted pages
    chunks = chunk_documents(all_pages, chunk_size=800, chunk_overlap=150)
    assert len(chunks) >= len(all_pages), "Total chunks should be greater than or equal to source pages."

    for idx, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["source"].endswith(".pdf")
        assert len(chunk.page_content.strip()) > 0

from typing import List, Dict, Any
from app.services.vector_store import similarity_search_with_metadata

GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "id": 1,
        "query": "What is Nubank's customer base size and activity rate in Brazil?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["110 million", "85%"],
    },
    {
        "id": 2,
        "query": "What are the customer growth milestones in Mexico and Colombia?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["13 million", "4 million"],
    },
    {
        "id": 3,
        "query": "What was Nubank's total credit portfolio size and growth rate?",
        "expected_quarter": "3Q25",
        "expected_page": 1,
        "expected_keywords": ["30.4 billion", "42%"],
    },
    {
        "id": 4,
        "query": "What were the business highlights and new product launches across markets?",
        "expected_quarter": "4Q25",
        "expected_page": 3,
        "expected_keywords": ["100 new products", "payroll loan"],
    },
    {
        "id": 5,
        "query": "How is Nubank positioned relative to private financial institutions in Brazil?",
        "expected_quarter": "4Q25",
        "expected_page": 1,
        "expected_keywords": ["largest private", "Central Bank"],
    },
]


# ==========================================
# 4.4 Golden Dataset Retrieval Benchmark
# ==========================================

def test_retrieval_golden_dataset_hit_rate():
    """Benchmarks semantic search against the Golden Dataset; asserts top-4 hit rate >= 80%."""
    k = 4
    hits = 0
    total = len(GOLDEN_DATASET)

    for item in GOLDEN_DATASET:
        query = item["query"]
        expected_page = item["expected_page"]
        expected_quarter = item["expected_quarter"]

        retrieved_docs = similarity_search_with_metadata(query=query, k=k)
        assert len(retrieved_docs) > 0, f"Retrieval returned 0 chunks for query: '{query}'"

        hit_found = any(
            doc.metadata.get("page") == expected_page and doc.metadata.get("fiscal_period") == expected_quarter
            for doc in retrieved_docs
        )

        if hit_found:
            hits += 1

    hit_rate = (hits / total) * 100
    assert hit_rate >= 80.0, f"Hit rate dropped below 80%! Got {hit_rate:.1f}% ({hits}/{total})"


def test_similarity_search_metadata_filter():
    """Verifies that post-retrieval metadata filtering strictly isolates the target quarter."""
    query = "What is the total revenue and credit portfolio growth?"
    filter_period = "3Q25"

    results = similarity_search_with_metadata(
        query=query,
        k=3,
        filter_dict={"fiscal_period": filter_period},
    )

    assert len(results) > 0
    for doc in results:
        assert doc.metadata.get("fiscal_period") == filter_period