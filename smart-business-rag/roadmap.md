# 🗺️ Roadmap: Smart Business Operations RAG Assistant

An enterprise-oriented Decision Support System designed to analyze unstructured corporate filings (e.g., 10-Ks, quarterly earnings, operational guidelines) and return strictly verified, cited operational insights.

---

## 🛠️ The Tech Stack (Zero-Dollar / "Crusty Nails")
* **Language & Runtime:** Python 3.11+
* **Package / Environment Manager:** `uv`
* **Synthesis & Reasoning LLM:** Google Gemini (`gemini-2.5-flash` / `gemini-1.5-flash`) via `langchain-google-genai`
* **Fast Routing & Grading LLM:** Groq (`openai/gpt-oss-120b` or current high-speed tier) via `langchain-groq`
* **Embeddings:** Google AI Studio (`models/text-embedding-004`) or local `sentence-transformers`
* **Vector Store & Relational DB:** PostgreSQL with `pgvector` extension (Docker local or Supabase Free Tier)
* **API & App Framework:** FastAPI, Pydantic v2, SQLModel / SQLAlchemy
* **Agent Orchestration & Workflow:** LangGraph, LangChain
* **Observability & Tracing:** LangSmith
* **Quality Assurance & DevOps:** Pytest, Docker, Docker Compose

---

## 📅 Multi-Part Action Plan

### 📋 Phase 0: Groundwork & Environment Validation [COMPLETED ✅]
- [x] Initialize repo and package management using `uv`.
- [x] Configure centralized `.env` secrets.
- [x] Validate model connectivity (`test_connections.py`) for Gemini and Groq.
- [x] Choose sample corporate filings dataset (e.g., 2 consecutive quarterly reports from Nubank, Petrobras, or Mercado Livre).

---

### 📥 Phase 1: Ingestion, Metadata Enrichment & Vector Pipeline
> **Goal:** Transform unstructured PDFs into cleanly chunked, highly searchable vector embeddings with rigorous page-level provenance.

- [ ] **1.1 PDF Extraction:**
  - Build an ingestion module using `pypdf` that extracts text page-by-page.
  - Retain structural metadata: `document_name`, `company`, `fiscal_period`, and `page_number`.
- [ ] **1.2 Chunking Strategy:**
  - Implement `RecursiveCharacterTextSplitter` with tuned chunk size (~800–1000 tokens) and overlap (~150 tokens) to preserve financial table context.
- [ ] **1.3 Vector Store Setup (`pgvector`):**
  - Spin up PostgreSQL with `pgvector` (via local `docker-compose.yml` or remote Supabase).
  - Define the vector schema using SQLModel/SQLAlchemy.
- [ ] **1.4 Embedding & Loading:**
  - Generate embeddings and persist chunks with metadata into the vector database.
  - Write an isolated smoke test script to verify semantic similarity search returns relevant chunks with correct metadata.

---

### 🧠 Phase 2: Agentic RAG Workflow with LangGraph
> **Goal:** Implement a self-correcting retrieval graph to eliminate hallucinations and enforce strict business citations.

- [ ] **2.1 Define the Graph State:**
  - Create a Pydantic/TypedDict state holding: `query`, `documents`, `web_search_fallback`, `generation`, and `retry_count`.
- [ ] **2.2 Fast Node - Retrieval Grader (Groq):**
  - Use Groq for ultra-low latency relevance grading.
  - Prompt: Determine whether retrieved chunks contain sufficient context to answer the business question.
- [ ] **2.3 Heavy Node - Answer Synthesis & Citations (Gemini):**
  - Pass validated chunks to Gemini with a strict system prompt.
  - Enforce structured output via Pydantic:
    ```json
    {
      "answer": "Net margin increased by 2.4%...",
      "citations": [
        {"document": "Q3_Report.pdf", "page": 12, "quote": "..."}
      ]
    }
    ```
- [ ] **2.4 Guardrail Node - Hallucination / Grounding Check:**
  - Ensure every financial/operational metric in the final answer is directly supported by the retrieved quotes.
- [ ] **2.5 Observability:**
  - Verify LangSmith project dashboard logs every node transition, latency profile, and token count.

---

### 🚀 Phase 3: The Production API Layer (FastAPI)
> **Goal:** Expose the ingestion and RAG agent pipelines via robust, documented REST endpoints.

- [ ] **3.1 Architecture & Project Structure:**
  - Set up a clean layered pattern:
    ```text
    app/
    ├── api/v1/endpoints/  # Route handlers (/query, /documents, /health)
    ├── core/              # Config, settings, logging
    ├── models/            # Pydantic schemas (Request/Response models)
    ├── services/          # LangGraph graph, vector store client
    └── main.py            # FastAPI entry point & CORS
    ```
- [ ] **3.2 Ingestion Endpoint (`POST /api/v1/documents/upload`):**
  - Accept PDF file uploads, execute the ingestion pipeline asynchronously, and return an ingestion summary status.
- [ ] **3.3 Operational Query Endpoint (`POST /api/v1/query`):**
  - Accept query payload with optional metadata filters (e.g., `company`, `year`).
  - Stream or return the structured response with citations.
- [ ] **3.4 Health & Metrics (`GET /health`):**
  - Verify DB connection and LLM API availability for container health checks.

---

### 🛡️ Phase 4: Automated Testing & Verification
> **Goal:** Guarantee production reliability and prevent regressions.

- [ ] **4.1 Unit Testing:**
  - Test chunking logic, metadata preservation, and Pydantic schema validation.
- [ ] **4.2 Integration Testing:**
  - Write Pytest routines mocking external LLM responses to test FastAPI endpoints and status codes (200, 422, 500).
- [ ] **4.3 Retrieval Evaluation:**
  - Create a golden dataset of 5 representative business questions with known ground-truth citations to benchmark retrieval accuracy.

---

### 🐳 Phase 5: Containerization, Deployment & Documentation
> **Goal:** Deliver a showcase-ready repository that any hiring manager or tech lead can clone and run in one command.

- [ ] **5.1 Dockerization:**
  - Write a multi-stage `Dockerfile` optimizing build time and image size with `uv`.
  - Write a `docker-compose.yml` linking the FastAPI app and the `pgvector` container.
- [ ] **5.2 Cloud Deployment (Free Tier):**
  - Deploy FastAPI service to **Render** (or Railway).
  - Connect to **Supabase** for the hosted vector database.
- [ ] **5.3 Portfolio Readme & Video Walkthrough:**
  - Write an executive README: Business context, architecture diagram, instructions to run locally, sample API calls, and LangSmith tracing screenshots.

---

## 🎯 Progress Tracker
- [x] Phase 0: Foundations & Connections
- [ ] Phase 1: Ingestion & Vector Storage
- [ ] Phase 2: LangGraph Agentic Pipeline
- [ ] Phase 3: FastAPI Backend
- [ ] Phase 4: Pytest Suite
- [ ] Phase 5: Docker & Cloud Deployment