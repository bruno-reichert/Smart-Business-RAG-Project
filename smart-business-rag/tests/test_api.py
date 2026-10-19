from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.schemas import OperationalInsight, Citation

client = TestClient(app)


def test_root_endpoint():
    """GET / should return 200 OK with welcome message and docs URL."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint():
    """GET /api/v1/health should return 200 OK and valid health schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database_connected" in data
    assert "version" in data


def test_query_validation_error():
    """POST /api/v1/filings/query with a 2-letter query should fail with HTTP 422."""
    payload = {"query": "ab"}  # Fails min_length=3 rule
    response = client.post("/api/v1/filings/query", json=payload)
    assert response.status_code == 422
    errors = response.json()
    assert "detail" in errors


@patch("app.api.v1.endpoints.query.rag_agent.invoke")
def test_query_success_with_mocked_agent(mock_invoke):
    """
    POST /api/v1/filings/query should return 200 OK with structured citations.
    Uses a mock so this test runs instantly, offline, and spends 0 API tokens.
    """
    # 1. Configure the simulated agent return value
    mock_insight = OperationalInsight(
        answer="Nubank's customer base reached 110 million in Brazil.",
        citations=[
            Citation(
                document="nubank-3q25.pdf",
                page=1,
                quote="With a customer base reaching 110 million"
            )
        ],
        is_grounded=True
    )
    mock_invoke.return_value = {"final_insight": mock_insight}

    # 2. Make the HTTP request
    payload = {
        "query": "What is Nubank's customer count?",
        "fiscal_filter": "3Q25"
    }
    response = client.post("/api/v1/filings/query", json=payload)

    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["fiscal_filter"] == "3Q25"
    assert data["insight"]["is_grounded"] is True
    assert len(data["insight"]["citations"]) == 1
    assert data["insight"]["citations"][0]["page"] == 1