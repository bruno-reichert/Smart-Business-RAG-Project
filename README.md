# 💼 Smart Business Operations RAG Assistant
### Grounded Multi-Turn Decision Support System for Corporate & Financial Filings

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![LangSmith Evaluated](https://img.shields.io/badge/Evaluated%20with-LangSmith-green.svg)](https://smith.langchain.com/)
[![pgvector](https://img.shields.io/badge/VectorDB-PostgreSQL%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-blue.svg)](https://www.docker.com/)

An enterprise-oriented Retrieval-Augmented Generation (RAG) assistant designed to solve a core business bottleneck: **analyzing dense, unstructured corporate filings (e.g., quarterly reports, 10-Ks, shareholder releases) while eliminating hallucinations and enforcing strict, page-level citation provenance.**

---

## 🏛️ System Architecture

The system decouples fast micro-evaluations from high-accuracy reasoning using a **Dual-Engine / Zero-Dollar inference architecture**:

```text
       [ User Query ] (e.g. "What about Colombia?")
              │
              ▼
   ┌───────────────────────┐
   │ 0. QUERY REFORMULATOR │  <-- Groq (sub-200ms) re-writes follow-ups
   │   (Short-Term Memory) │      into standalone search queries
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  1. HYBRID RETRIEVAL  │  <-- PostgreSQL + pgvector (Supabase)
   │  (Metadata Filtering) │      Searches top-k relevant chunks
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  2. RELEVANCE GRADER  │  <-- Groq micro-evaluator discards
   │    (Fast Guardrail)   │      irrelevant noise & hallucinations
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
