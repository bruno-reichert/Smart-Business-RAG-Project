from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Decision Support System powered by LangGraph, pgvector, and Dual-LLM (Groq + Gemini).",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration (allows frontend clients to connect cleanly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all V1 API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Smart Business Operations RAG API is operational.",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health"
    }