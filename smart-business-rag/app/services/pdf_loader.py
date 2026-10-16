import os
import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Infers company and fiscal period from conventional filenames
    e.g., 'nubank-3q25.pdf' -> {'company': 'Nubank', 'fiscal_period': '3Q25'}
    """
    clean_name = Path(filename).stem.lower()
    
    # Default values
    metadata = {
        "company": "Nubank",
        "fiscal_period": "Unknown",
        "source_file": Path(filename).name
    }
    
    # Extract quarter pattern like 1q24, 3q25, 4q23
    quarter_match = re.search(r"(\d[qQ]\d{2})", clean_name)
    if quarter_match:
        metadata["fiscal_period"] = quarter_match.group(1).upper()
        
    if "nubank" in clean_name or "nu" in clean_name:
        metadata["company"] = "Nubank"
        
    return metadata

def clean_text(text: str) -> str:
    """
    Basic text sanitation: handles erratic whitespace from PDF extractions.
    """
    if not text:
        return ""
    # Replace multiple consecutive spaces/newlines with single ones
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def load_pdf_with_metadata(file_path: str) -> List[Document]:
    """
    Extracts text page by page from a PDF and returns LangChain Document objects
    enriched with page numbers and document-level metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    doc_metadata = extract_metadata_from_filename(path.name)
    reader = PdfReader(str(path))
    documents: List[Document] = []
    
    print(f"Loading '{path.name}' ({len(reader.pages)} pages)...")
    
    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned_page_text = clean_text(raw_text)
        
        # Skip empty pages (e.g. blank spacer pages)
        if not cleaned_page_text:
            continue
            
        page_metadata: Dict[str, Any] = {
            "source": path.name,
            "company": doc_metadata["company"],
            "fiscal_period": doc_metadata["fiscal_period"],
            "page": page_idx + 1,  # 1-indexed for human readability
            "total_pages": len(reader.pages)
        }
        
        documents.append(
            Document(page_content=cleaned_page_text, metadata=page_metadata)
        )
        
    print(f"-> Extracted {len(documents)} text pages from {path.name}.")
    return documents