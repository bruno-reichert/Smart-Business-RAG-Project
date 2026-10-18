from fastapi import APIRouter
from app.api.v1.endpoints import health, query, documents

api_router = APIRouter()

api_router.include_router(health.router, tags=["System Health"])
api_router.include_router(query.router, prefix="/filings", tags=["Agentic RAG Query"])
api_router.include_router(documents.router, prefix="/documents", tags=["Document Ingestion"])