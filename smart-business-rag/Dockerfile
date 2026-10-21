# Use official lightweight Python image
FROM python:3.11-slim-bookworm

# Install uv directly from Astral's official binary distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation & copy environment efficiently
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency definition files first for optimal Docker layer caching
COPY pyproject.toml uv.lock* ./

# Install project dependencies using uv sync
RUN uv sync --frozen --no-install-project

# Copy application source code
COPY app/ ./app/
COPY data/ ./data/

# Place the uv virtual environment at the beginning of PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default port exposed for FastAPI backend
EXPOSE 8000

# Default command launches FastAPI
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]