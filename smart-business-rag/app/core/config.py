from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Business Operations RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Keys & Secrets - Defaulting to empty str or Field(...) satisfies Pylance
    GOOGLE_API_KEY: str = Field(default="", description="Google AI Studio API Key")
    GROQ_API_KEY: str = Field(default="", description="Groq API Key")
    DATABASE_URL: str = Field(default="", description="PostgreSQL Connection URI")
    
    # LangSmith Observability
    LANGCHAIN_TRACING_V2: Optional[str] = "true"
    LANGCHAIN_ENDPOINT: Optional[str] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = "smart-business-rag-v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()