from pathlib import Path
from app.services.pdf_loader import load_pdf_with_metadata
from app.services.chunker import chunk_documents

def main():
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/ directory!")
        return
        
    all_pages = []
    for pdf_path in pdf_files:
        pages = load_pdf_with_metadata(str(pdf_path))
        all_pages.extend(pages)
        
    print(f"\n--- Total pages extracted: {len(all_pages)} ---")
    
    # Run the chunker
    chunks = chunk_documents(all_pages, chunk_size=800, chunk_overlap=150)
    
    print("\n--- SAMPLE CHUNK (Index 0) ---")
    print("Metadata:", chunks[0].metadata)
    print("Content:\n" + chunks[0].page_content)
    print("-" * 50)
    
    print("\n--- SAMPLE CHUNK (Index 1) ---")
    print("Metadata:", chunks[1].metadata)
    print("Content:\n" + chunks[1].page_content)

if __name__ == "__main__":
    main()