import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

_db_url = os.getenv("DATABASE_URL")
if not _db_url:
    raise ValueError("DATABASE_URL is missing in .env")

DATABASE_URL: str = _db_url
COLLECTION_NAME = "nubank_filings_collection"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes a lightweight local embedding model.
    Runs locally on your CPU with zero API calls and zero rate limits.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store() -> PGVector:
    """Initializes PGVector connecting to Supabase PostgreSQL."""
    embeddings = get_embeddings_model()
    
    return PGVector(
        connection_string=DATABASE_URL,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )

def index_documents(chunks: List[Document]) -> None:
    """Stores and embeds a list of document chunks into Supabase."""
    vector_store = get_vector_store()
    print(f"Embedding and uploading {len(chunks)} chunks to Supabase...")
    vector_store.add_documents(chunks)
    print("Indexing complete! All embeddings are stored in Supabase.")

def similarity_search_with_metadata(
    query: str, 
    k: int = 4, 
    filter_dict: Optional[dict] = None
) -> List[Document]:
    """
    Executes a semantic similarity search with optional metadata filtering.
    e.g., filter_dict={"fiscal_period": "3Q25"}
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k, filter=filter_dict)