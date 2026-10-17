from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from app.models.schemas import OperationalInsight

class AgentState(TypedDict):
    """
    Represents the internal state of the LangGraph workflow.
    Each node receives this state, modifies or appends keys, and passes it forward.
    """
    query: str                               # Original question asked by the user
    fiscal_filter: Optional[str]             # Optional metadata filter (e.g., '3Q25')
    documents: List[Document]                # Chunks retrieved from Supabase pgvector
    relevant_documents: List[Document]       # Chunks that passed Groq's relevance test
    final_insight: Optional[OperationalInsight] # The final structured output