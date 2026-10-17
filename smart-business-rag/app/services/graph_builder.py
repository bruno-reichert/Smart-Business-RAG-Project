from typing import Literal
from langgraph.graph import StateGraph, START, END

from app.services.agent_state import AgentState
from app.services.rag_nodes import (
    retrieve_node,
    grade_documents_node,
    generate_node,
    fallback_node,
)


def decide_to_generate(state: AgentState) -> Literal["generate", "fallback"]:
    """
    Conditional Edge / Guardrail:
    Inspects if we have any relevant documents after Groq's evaluation.
    If yes -> proceeds to Gemini for synthesis with citations.
    If no  -> routes to fallback to prevent hallucinating metrics.
    """
    relevant_docs = state.get("relevant_documents", [])
    
    if not relevant_docs:
        print("[Router] Decision: 0 relevant documents found -> Routing to FALLBACK.")
        return "fallback"
    
    print(f"[Router] Decision: {len(relevant_docs)} relevant document(s) verified -> Routing to GENERATE.")
    return "generate"


def build_rag_graph():
    """Compiles the agentic RAG workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add all functional nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # 2. Add sequential flow: START -> retrieve -> grade_documents
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 3. Add conditional edge (Guardrail) from grade_documents
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    # 4. Both terminal nodes lead to END
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Compile the graph into an executable Runnable
    app = workflow.compile()
    return app