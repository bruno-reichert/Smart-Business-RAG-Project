from fastapi import APIRouter, status
from sqlalchemy import create_engine, text
from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def check_health():
    """Verifies that the API service and Supabase PostgreSQL connection are alive."""
    db_alive = False
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
            db_alive = True
    except Exception as e:
        print(f"Health check DB error: {e}")

    return HealthResponse(
        status="healthy" if db_alive else "degraded",
        database_connected=db_alive,
        version=settings.VERSION
    )