import pytest
from app.services.graph_builder import build_rag_graph
from app.services.agent_state import AgentState

# Mark all agent tests in this file as live integration tests
pytestmark = pytest.mark.integration


# ==========================================
# 5.1 Single-Turn Grounded Query Test
# ==========================================

def test_agent_grounded_operational_query():
    """Executes a valid operational query through LangGraph; verifies grounded answer and citations."""
    agent = build_rag_graph()

    state_input: AgentState = {
        "query": "What are Nubank's customer growth milestones in Mexico and Colombia in 3Q25?",
        "fiscal_filter": "3Q25",
    }
    config = {"configurable": {"thread_id": "test-grounded-session"}}

    result = agent.invoke(state_input, config=config)
    insight = result.get("final_insight")

    assert insight is not None, "Workflow failed to produce final_insight."
    assert insight.is_grounded is True, "Answer was expected to be grounded in filings."
    assert len(insight.answer.strip()) > 30, "Generated answer is too short."

    # Verify structured citations
    assert len(insight.citations) > 0, "Expected at least 1 citation for grounded query."
    for citation in insight.citations:
        assert citation.document.endswith(".pdf")
        assert citation.page >= 1
        assert len(citation.quote.strip()) > 0

# ==========================================
# 5.2 Out-of-Scope Fallback Guardrail Test
# ==========================================

def test_agent_fallback_on_irrelevant_query():
    """Asks an out-of-scope question; verifies Groq discards noise and routes to fallback."""
    agent = build_rag_graph()

    state_input: AgentState = {
        "query": "What are the engineering specs of the rocket engine used by SpaceX to reach Mars?",
        "fiscal_filter": None,
    }
    config = {"configurable": {"thread_id": "test-fallback-session"}}

    result = agent.invoke(state_input, config=config)
    insight = result.get("final_insight")

    assert insight is not None, "Workflow failed to produce final_insight."
    assert insight.is_grounded is False, "Irrelevant query must NOT be marked as grounded."
    assert len(insight.citations) == 0, "Fallback responses must contain 0 citations."
    assert "no verifiable information" in insight.answer.lower()

# ==========================================
# 5.3 Multi-Turn Conversational Memory Test
# ==========================================

def test_agent_multi_turn_conversational_memory():
    """Executes a 2-turn conversation sharing the same thread_id; tests Groq reformulation & memory."""
    agent = build_rag_graph()
    session_config = {"configurable": {"thread_id": "test-multiturn-memory-session"}}

    # --- TURN 1 ---
    q1 = "What are Nubank's customer growth milestones in Mexico?"
    input_1: AgentState = {"query": q1, "fiscal_filter": None}

    res1 = agent.invoke(input_1, config=session_config)
    insight_1 = res1.get("final_insight")

    assert insight_1 is not None
    assert insight_1.is_grounded is True
    assert "mexico" in insight_1.answer.lower()

    # --- TURN 2 (Follow-up relying on Turn 1 context) ---
    q2 = "And what about Colombia?"
    input_2: AgentState = {"query": q2, "fiscal_filter": None}

    res2 = agent.invoke(input_2, config=session_config)
    insight_2 = res2.get("final_insight")

    assert insight_2 is not None
    assert insight_2.is_grounded is True
    assert "colombia" in insight_2.answer.lower()

    # Verify conversation state accumulation across turns
    messages = res2.get("messages", [])
    assert len(messages) >= 4, f"Expected at least 4 messages across 2 turns, got {len(messages)}"