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
    Executes a semantic similarity search.
    If a filter_dict is provided (e.g., {"fiscal_period": "4Q25"}),
    it retrieves candidate chunks and filters them reliably without
    triggering Postgres JSONPath version mismatches.
    """
    vector_store = get_vector_store()
    
    if not filter_dict:
        # Standard search without filters
        return vector_store.similarity_search(query, k=k)
    
    # Retrieve slightly more candidates to account for filtering
    raw_results = vector_store.similarity_search(query, k=k * 3)
    
    # Filter candidates by matching metadata keys
    filtered_results: List[Document] = []
    for doc in raw_results:
        matches = True
        for key, expected_value in filter_dict.items():
            if str(doc.metadata.get(key, "")).upper() != str(expected_value).upper():
                matches = False
                break
        if matches:
            filtered_results.append(doc)
        if len(filtered_results) >= k:
            break
            
    # If the filter was too restrictive and returned nothing, fallback to raw top results
    return filtered_results if filtered_results else raw_results[:k]