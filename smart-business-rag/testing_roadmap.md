# 🧪 Test Suite Revamp Roadmap

## Step 1: Pytest Setup & Fixtures
- [X] **1.1 Configure Pytest Settings:** Add pytest configuration and custom markers (`integration`) to `pyproject.toml`.
- [X] **1.2 Shared Fixtures:** Create `tests_new/conftest.py` for shared fixtures (mock data, environment setup, and helper factories).

## Step 2: Unit Tests (`tests_new/test_unit.py`)
- [X] **2.1 PDF Loader & Sanitation:**
  - [X] Test filename pattern parsing (company, fiscal period extraction).
  - [X] Test text cleaner (whitespace normalization, edge cases like empty strings).
- [X] **2.2 Chunker Mechanics:**
  - [X] Test document splitting with overlapping windows.
  - [X] Verify parent metadata inheritance and deterministic `chunk_index` assignment.
- [X] **2.3 Pydantic Schemas & Contracts:**
  - [X] Test valid and invalid payloads for `QueryRequest` (e.g., query length constraints).
  - [X] Test structured output serialization for `OperationalInsight` and `Citation`.
- [X] **2.4 LangGraph Isolated Router Guardrail:**
  - [X] Unit test `decide_to_generate` conditional router (empty vs. populated relevant documents) without invoking LLMs.

## Step 3: Mocked API Integration Tests (`tests_new/test_api.py`)
- [X] **3.1 System Endpoints:** Test `/` (root) and `/api/v1/health` status codes and payload structures.
- [X] **3.2 Input Validation:** Test bad requests on `/api/v1/filings/query` (HTTP 422 triggers).
- [X] **3.3 Mocked Query Execution:** Test happy path for conversational query endpoint with mocked agent response.
- [x] **3.4 Ingestion Endpoint:** Test `/api/v1/documents/upload` validation (reject non-PDFs) and mocked successful upload.

## Step 4: Live Infrastructure Tests (`tests_new/test_live_integrations.py` - `@pytest.mark.integration`)
- [ ] **4.1 Model Connectivity:** Live ping to Groq (`openai/gpt-oss-120b`) and Gemini (`gemini-3.5-flash`).
- [ ] **4.2 Database Connectivity:** Live connection to Supabase PostgreSQL and verify `pgvector` extension activation.
- [ ] **4.3 Real Document Ingestion:** Verify page extraction and chunking against actual files in `data/`.
- [ ] **4.4 Golden Dataset Retrieval Benchmark:** Live pgvector retrieval benchmark validating top-k page provenance and hit rate.

## Step 5: Live LangGraph Workflow Tests (`tests_new/test_live_agent.py` - `@pytest.mark.integration`)
- [ ] **5.1 Single-Turn Grounded Query:** Ask a valid corporate filing question; verify `is_grounded=True` and citations structure.
- [ ] **5.2 Fallback / Guardrail Routing:** Ask an out-of-scope question; verify routing to `fallback` node (`is_grounded=False`).
- [ ] **5.3 Multi-Turn Memory & Reformulation:** Execute a two-turn conversation using `thread_id` to verify context retention across follow-up queries.

## Step 6: Verification & Final Migration
- [ ] **6.1 Offline Verification:** Run unit test suite (`uv run pytest -m "not integration"`).
- [ ] **6.2 Live Verification:** Run full test suite with live services (`uv run pytest -m "integration"`).
- [ ] **6.3 Final Migration:** Deprecate old `tests/` folder and promote `tests_new/` to `tests/`.