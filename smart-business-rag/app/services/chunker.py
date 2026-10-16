from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 150) -> RecursiveCharacterTextSplitter:
    """
    Creates a text splitter configured for narrative and semi-structured business text.
    It splits by double newlines (paragraphs) first, then single newlines, then spaces.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    Splits a list of page-level Documents into smaller chunk-level Documents,
    automatically preserving parent metadata and appending chunk index tracking.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    # Enrich chunks with a unique chunk ID relative to its document
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        
    print(f"Split {len(documents)} source pages into {len(chunks)} searchable chunks.")
    return chunks