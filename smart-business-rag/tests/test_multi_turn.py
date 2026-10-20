from dotenv import load_dotenv
from app.services.graph_builder import build_rag_graph
from app.services.agent_state import AgentState
from langchain_core.runnables import RunnableConfig


load_dotenv()

def main():
    agent = build_rag_graph()
    session_config: RunnableConfig = {"configurable": {"thread_id": "test-session-42"}}

    print("=== TURN 1 ===")
    q1 = "What is Nubank's customer growth in Mexico in 4Q25?"
    input_1: AgentState = {"query": q1, "fiscal_filter": "4Q25"}
    res1 = agent.invoke(input_1, config=session_config)
    print(f"A1: {res1['final_insight'].answer[:150]}...\n")

    print("=== TURN 2 (Follow-up with pronoun / missing context) ===")
    q2 = "And what about Colombia in that same period?"
    input_2: AgentState = {"query": q2, "fiscal_filter": "4Q25"}
    res2 = agent.invoke(input_2, config=session_config)
    print(f"A2: {res2['final_insight'].answer[:150]}...\n")

if __name__ == "__main__":
    main()