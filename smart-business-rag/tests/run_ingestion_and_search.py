from pathlib import Path
from app.services.pdf_loader import load_pdf_with_metadata
from app.services.chunker import chunk_documents
from app.services.vector_store import index_documents, similarity_search_with_metadata

def main():
    print("=== Step 1: Loading PDFs ===")
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    all_pages = []
    for pdf_path in pdf_files:
        pages = load_pdf_with_metadata(str(pdf_path))
        all_pages.extend(pages)
        
    print(f"Total pages: {len(all_pages)}")

    print("\n=== Step 2: Splitting into Chunks ===")
    chunks = chunk_documents(all_pages, chunk_size=800, chunk_overlap=150)

    print("\n=== Step 3: Indexing into Supabase pgvector ===")
    index_documents(chunks)

    print("\n=== Step 4: Testing Semantic Retrieval ===")
    test_query = "What is Nubank's customer growth and activity rate in Brazil?"
    print(f"Query: '{test_query}'\n")
    
    results = similarity_search_with_metadata(test_query, k=2)
    
    for i, doc in enumerate(results):
        print(f"--- MATCH {i + 1} ---")
        print(f"Source: {doc.metadata.get('source')} | Quarter: {doc.metadata.get('fiscal_period')} | Page: {doc.metadata.get('page')}")
        print("Content preview:")
        print(doc.page_content[:250] + "...\n")

if __name__ == "__main__":
    main()