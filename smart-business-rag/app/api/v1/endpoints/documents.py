import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.models.schemas import DocumentUploadResponse
from app.services.pdf_loader import load_pdf_with_metadata
from app.services.chunker import chunk_documents
from app.services.vector_store import index_documents

router = APIRouter()
UPLOAD_DIR = Path("data")

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a new quarterly corporate PDF, extracts its text page-by-page,
    chunks it, and indexes vector embeddings directly into Supabase.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    # Save uploaded file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}"
        )

    # Process through pipeline
    try:
        pages = load_pdf_with_metadata(str(file_path))
        chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=150)
        index_documents(chunks)
        
        return DocumentUploadResponse(
            filename=file.filename,
            pages_extracted=len(pages),
            chunks_indexed=len(chunks),
            message="Document successfully processed and indexed into vector store."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {e}"
        )