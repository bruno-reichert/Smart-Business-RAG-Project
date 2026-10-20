from typing import List, Optional, Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from app.models.schemas import OperationalInsight

class AgentState(TypedDict, total=False):
    """
    Represents the internal state of the conversational LangGraph workflow.
    `messages` uses `add_messages` to automatically append new conversation turns.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages] # Multi-turn conversation history
    query: str                                              # Current user question (or reformulated query)
    fiscal_filter: Optional[str]                            # Metadata filter (e.g., '3Q25')
    documents: List[Document]                               # Chunks retrieved from pgvector
    relevant_documents: List[Document]                      # Chunks that passed Groq's relevance test
    final_insight: Optional[OperationalInsight]                # Synthesized output with citations