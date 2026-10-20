from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.agent_state import AgentState
from app.services.rag_nodes import (
    reformulate_query_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    fallback_node,
)


def decide_to_generate(state: AgentState) -> Literal["generate", "fallback"]:
    """Conditional Edge / Guardrail based on verified document relevance."""
    relevant_docs = state.get("relevant_documents", [])
    
    if not relevant_docs:
        print("[Router] Decision: 0 relevant documents found -> Routing to FALLBACK.")
        return "fallback"
    
    print(f"[Router] Decision: {len(relevant_docs)} relevant document(s) verified -> Routing to GENERATE.")
    return "generate"


def build_rag_graph():
    """Compiles the conversational RAG workflow with state checkpointing."""
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("reformulate_query", reformulate_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # 2. Sequential flow: START -> reformulate_query -> retrieve -> grade_documents
    workflow.add_edge(START, "reformulate_query")
    workflow.add_edge("reformulate_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 3. Conditional routing from grade_documents
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    # 4. Terminal edges
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # 5. Attach in-memory checkpointer for thread persistence
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app