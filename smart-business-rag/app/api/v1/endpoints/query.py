from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_state import AgentState
from app.services.graph_builder import build_rag_graph

router = APIRouter()

# Compile the LangGraph agent once during startup so it's ready for requests
rag_agent = build_rag_graph()

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_filings(request: QueryRequest):
    """
    Submits a business query to the Agentic RAG workflow.
    Evaluates context via Groq, synthesizes insights via Gemini, and enforces citations.
    """
    try:
        initial_state: AgentState = {
            "query": request.query,
            "fiscal_filter": request.fiscal_filter,
            "documents": [],
            "relevant_documents": [],
            "final_insight": None
        }
        
        # Invoke our LangGraph pipeline
        result = rag_agent.invoke(initial_state)
        insight = result.get("final_insight")
        
        if not insight:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow failed to generate an insight."
            )
            
        return QueryResponse(
            query=request.query,
            fiscal_filter=request.fiscal_filter,
            insight=insight
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent query: {str(e)}"
        )