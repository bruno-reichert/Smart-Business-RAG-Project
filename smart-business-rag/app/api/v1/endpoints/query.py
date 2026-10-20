import uuid
from fastapi import APIRouter, HTTPException, status
from langchain_core.runnables import RunnableConfig

from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_state import AgentState
from app.services.graph_builder import build_rag_graph

router = APIRouter()

# Compile the LangGraph agent once during startup with checkpointer enabled
rag_agent = build_rag_graph()

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_filings(request: QueryRequest):
    """
    Submits a business query to the Conversational Agentic RAG workflow.
    Tracks session history across turns via thread_id, reformulates follow-ups,
    and returns verified citations.
    """
    try:
        # 1. Reuse existing thread_id or generate a fresh UUID for new conversations
        thread_id = request.thread_id or str(uuid.uuid4())
        session_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 2. Package state payload
        state_input: AgentState = {
            "query": request.query,
            "fiscal_filter": request.fiscal_filter
        }
        
        # 3. Invoke our LangGraph pipeline with session memory
        result = rag_agent.invoke(state_input, config=session_config)
        insight = result.get("final_insight")
        
        if not insight:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow failed to generate an insight."
            )
            
        return QueryResponse(
            query=request.query,
            fiscal_filter=request.fiscal_filter,
            thread_id=thread_id,
            insight=insight
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent query: {str(e)}"
        )