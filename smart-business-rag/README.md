# 💼 Smart Business Operations RAG Assistant
### Grounded Multi-Turn Decision Support System for Corporate & Financial Filings

[![Python 3.11+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![LangSmith Evaluated](https://img.shields.io/badge/Evaluated%20with-LangSmith-green.svg)](https://smith.langchain.com/)
[![pgvector](https://img.shields.io/badge/VectorDB-PostgreSQL%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-blue.svg)](https://www.docker.com/)

An enterprise-oriented Retrieval-Augmented Generation (RAG) assistant designed to solve a core business bottleneck: **analyzing dense, unstructured corporate filings (e.g., quarterly reports, 10-Ks, shareholder releases) while reducing hallucinations and requiring page-level citations.**

_Built with AI-assisted development (Google AI Studio); design decisions, integration, and debugging by the author._

---

## 🏛️ System Architecture

The system decouples fast micro-evaluations from high-accuracy reasoning using a **Dual-Engine / Zero-Dollar inference architecture**:

```text
       [ User Query ] (e.g. "What about Colombia?")
              │
              ▼
   ┌───────────────────────┐
   │ 0. QUERY REFORMULATOR │  <-- Groq re-writes follow-ups
   │   (Short-Term Memory) │      into standalone search queries
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  1. VECTOR RETRIEVAL  │  <-- PostgreSQL + pgvector (Supabase)
   │  (Post-Retrieval      │      Searches top-k relevant chunks
   │   Metadata Filtering) │      
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  2. RELEVANCE GRADER  │  <-- Groq micro-evaluator discards
   │ (Per-Chunk LLM Check) │      irrelevant chunks before generation
   └──────────┬────────────┘
              │
         Is context valid?
        ┌─────┴─────┐
       YES          NO
        │            │
        ▼            ▼
 ┌─────────────┐   ┌───────────────────────────┐
 │ 3. GENERATE │   │ 4. FALLBACK / OUT-OF-SCOPE│
 └──────┬──────┘   └─────────────┬─────────────┘
        │                        │
        │ Gemini (Heavy Node)    │ Graceful notification:
        │ Pydantic JSON Citations│ "Data not verified in filings"
        │                        │
        └────────────┬───────────┘
                     ▼
       [ Streamlit Executive UI ]
```

---

## ✨ Key Features & Engineering Highlights

* **Structured Page-Level Citations:** Each answer returns document, page, and excerpt in a validated schema. Automated verification of excerpts against retrieved text is not yet implemented.
* **Dual-Engine Model Routing:**
  * **Groq (`openai/gpt-oss-120b`):** Handles fast conversational query reformulation and document relevance grading.
  * **Google Gemini (`gemini-3.5-flash`):** Executes complex financial synthesis and schema-validated structured outputs (Pydantic).
* **Multi-Turn Conversational Memory:** Built with LangGraph checkpointers (`MemorySaver`), preserving context across turns via unique `thread_id` sessions (in-memory, resets on restart).
* **Testing & Observability:**
  * Automated **Pytest** test suite with mocked LLM calls for fast, zero-token local testing.
  * **Benchmark & Evaluation:** Small golden dataset (6 questions) with latency and token tracing.
* **Dynamic PDF Ingestion Pipeline:** On-the-fly PDF extraction (`pypdf`), recursive character-based chunking with metadata preservation (`RecursiveCharacterTextSplitter`), and local vector embeddingsia FastEmbed (`all-MiniLM-L6-v2`).

---

## 📊 Benchmark & Evaluation (LangSmith)

The system was evaluated against a **Golden Dataset** of representative corporate queries targeting Nubank quarterly filings (3Q25 and 4Q25, 2Q25 also indexed).

| Metric | Result | Benchmark Target |
| :--- | :--- | :--- |
| **Page-Level Retrieval Hit Rate** | **80.0% (4 out of 5)** | $\ge 80.0\%$ |
| **Automated Tests** | **13 / 13 Passed** | 100% Pass Rate |

## ⚠️ Known Limitations

* **Small evaluation set:** 6 golden questions and 5 retrieval cases. Results are indicative, not statistically robust.
* **Citation check is not automated:** the model generates excerpts; they are not yet verified against retrieved text.
* **Vector-only retrieval:** no keyword (BM25) component or reranker.
* **Text-only PDF parsing:** tables in earnings reports lose structure with `pypdf`.
* **Volatile memory:** `MemorySaver` is in-memory and resets on restart.
* **Metadata filter:** applied after retrieval and falls back to unfiltered results if none match.
* **Demo scope:** single company, no authentication on the upload endpoint.
* **No correctness evaluation:** The LangSmith evaluator checks that the response is flagged as grounded and includes citations. It does not assess answer correctness.

---

## 🚀 Quickstart & Setup

### Prerequisites
* Python 3.13+
* `uv` package manager (`pip install uv` or via Astral install script)
* Free API keys for:
  * Google AI Studio (`GOOGLE_API_KEY`)
  * Groq Console (`GROQ_API_KEY`)
  * Supabase PostgreSQL URL (`DATABASE_URL`)
  * LangSmith (`LANGSMITH_API_KEY` - optional, for tracing)

### 1. Clone & Configure
```bash
git clone https://github.com/bruno-reichert/smart-business-rag.git
cd smart-business-rag
cp .env.example .env  # Populate your credentials
```

### 2. Run Locally with `uv`
```bash
# 1. Install dependencies
uv sync

# 2. Start FastAPI Backend (Terminal 1)
uv run uvicorn app.main:app --reload --port 8000

# 3. Start Streamlit UI (Terminal 2)
uv run streamlit run app/frontend/ui.py
```
* **Interactive UI:** `http://localhost:8501`
* **Swagger API Docs:** `http://localhost:8000/docs`

_The three Nubank filings in data/ (2Q25–4Q25) are indexed automatically at startup. Additional PDFs can be uploaded through the UI or the /documents/upload endpoint._


### 3. Run with Docker Compose
```bash
docker compose up --build
```

### 4. Run Test Suite & Benchmark
```bash
# Run unit & API integration tests
uv run pytest -v

# Run local retrieval benchmark
uv run python -m tests.evaluate_retrieval

# Run LangSmith cloud evaluation experiment
uv run python -m tests.run_langsmith_eval
```

---

## 📁 Repository Structure

```text
smart-business-rag/
├── app/
│   ├── api/v1/             # REST endpoints (/health, /filings/query, /documents/upload)
│   ├── core/               # Pydantic Settings & environment config
│   ├── frontend/           # Streamlit conversational chat interface
│   ├── models/             # Pydantic schemas (Contracts, Citations, Insights)
│   └── services/           # LangGraph graph, state, chunker, vector store
├── data/                   # Source corporate PDFs
├── tests/                  # Unit tests, integration tests, and evaluation scripts
├── Dockerfile              # Single-stage container build
├── docker-compose.yml      # Container orchestration
└── pyproject.toml          # Project configuration & dependencies
```



---

## 👨‍💻 Author

**Bruno Benno Reichert**  
*Bachelor in Business Administration (UFRGS) | Junior AI Engineer & Python Back-end Developer*  
* [LinkedIn](https://linkedin.com/in/brunobennoreichert) • [GitHub](https://github.com/bruno-reichert) • [Portfolio](https://bruno-reichert.github.io/portfolio-page)
