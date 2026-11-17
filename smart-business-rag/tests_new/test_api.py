from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ==========================================
# 3.1 System Endpoints (/ and /health)
# ==========================================

def test_root_endpoint():
    """GET / should return 200 OK with operational message and docs URLs."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"
    assert "/health" in data["health_check"]


def test_health_endpoint():
    """GET /api/v1/health should return 200 OK and conform to HealthResponse schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "database_connected" in data
    assert isinstance(data["database_connected"], bool)
    assert "version" in data

# ==========================================
# 3.2 Input Validation Tests (HTTP 422)
# ==========================================

def test_query_validation_too_short():
    """POST /api/v1/filings/query with a 2-letter query should fail with HTTP 422."""
    payload = {"query": "ab"}
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure the error detail specifically mentions the query length constraint
    assert any("at least 3 characters" in err.get("msg", "") for err in data["detail"])


def test_query_validation_missing_query():
    """POST /api/v1/filings/query without 'query' field should fail with HTTP 422."""
    payload = {"fiscal_filter": "3Q25"}
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert any(err.get("loc") == ["body", "query"] for err in data["detail"])


def test_query_validation_invalid_type():
    """POST /api/v1/filings/query with non-string query should fail with HTTP 422."""
    payload = {"query": {"invalid": "object"}}
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 422

from unittest.mock import patch
from app.models.schemas import OperationalInsight


# ==========================================
# 3.3 Mocked Query Execution Tests (HTTP 200 & 500)
# ==========================================

@patch("app.api.v1.endpoints.query.rag_agent.invoke")
def test_query_success_with_explicit_thread_id(mock_invoke, sample_insight: OperationalInsight):
    """POST /api/v1/filings/query with explicit thread_id returns 200 and preserves thread."""
    mock_invoke.return_value = {"final_insight": sample_insight}

    payload = {
        "query": "What is Nubank's customer count in Brazil?",
        "fiscal_filter": "3Q25",
        "thread_id": "my-custom-session-123",
    }
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["fiscal_filter"] == "3Q25"
    assert data["thread_id"] == "my-custom-session-123"
    assert data["insight"]["is_grounded"] is True
    assert len(data["insight"]["citations"]) == 1
    assert data["insight"]["citations"][0]["document"] == "nubank-3q25.pdf"


@patch("app.api.v1.endpoints.query.rag_agent.invoke")
def test_query_success_generates_thread_id_when_omitted(mock_invoke, sample_insight: OperationalInsight):
    """POST /api/v1/filings/query without thread_id returns 200 and auto-generates a UUID thread_id."""
    mock_invoke.return_value = {"final_insight": sample_insight}

    payload = {"query": "What were the key achievements?"}
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] is not None
    assert len(data["thread_id"]) > 10  # Verifies a valid UUID string was created


@patch("app.api.v1.endpoints.query.rag_agent.invoke")
def test_query_agent_returns_no_insight_triggers_500(mock_invoke):
    """POST /api/v1/filings/query should return HTTP 500 if the agent returns empty state."""
    mock_invoke.return_value = {"final_insight": None}

    payload = {"query": "A question that produces an empty insight"}
    response = client.post("/api/v1/filings/query", json=payload)

    assert response.status_code == 500
    assert "Workflow failed to generate an insight" in response.json()["detail"]

import io
from pathlib import Path
from unittest.mock import MagicMock


# ==========================================
# 3.4 Ingestion Endpoint Tests (/documents/upload)
# ==========================================

def test_upload_document_rejects_non_pdf():
    """POST /api/v1/documents/upload should reject non-PDF files with HTTP 400."""
    non_pdf_file = {"file": ("report.txt", io.BytesIO(b"plain text content"), "text/plain")}
    response = client.post("/api/v1/documents/upload", files=non_pdf_file)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


@patch("app.api.v1.endpoints.documents.index_documents")
@patch("app.api.v1.endpoints.documents.chunk_documents")
@patch("app.api.v1.endpoints.documents.load_pdf_with_metadata")
@patch("shutil.copyfileobj")
def test_upload_document_success(mock_copy, mock_load, mock_chunk, mock_index):
    """POST /api/v1/documents/upload should return 201 and valid DocumentUploadResponse."""
    # Mock pipeline returns
    mock_load.return_value = [MagicMock(), MagicMock()]  # 2 simulated pages
    mock_chunk.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]  # 4 simulated chunks
    mock_index.return_value = None

    fake_pdf = {"file": ("nubank-test-report.pdf", io.BytesIO(b"%PDF-1.4 simulated"), "application/pdf")}

    try:
        response = client.post("/api/v1/documents/upload", files=fake_pdf)

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "nubank-test-report.pdf"
        assert data["pages_extracted"] == 2
        assert data["chunks_indexed"] == 4
        assert "successfully processed" in data["message"]
    finally:
        # Clean up any empty mock file artifact created during test
        test_file = Path("data") / "nubank-test-report.pdf"
        if test_file.exists():
            test_file.unlink()