import json
from dotenv import load_dotenv
from app.services.graph_builder import build_rag_graph
from app.services.agent_state import AgentState

load_dotenv()

def print_result(title: str, result: dict):
    print("\n" + "=" * 60)
    print(f"📌 {title.upper()}")
    print("=" * 60)
    
    insight = result.get("final_insight")
    if not insight:
        print("No insight generated.")
        return

    print(f"\n🧠 EXECUTIVE ANSWER:\n{insight.answer}\n")
    print(f"🛡️ GROUNDED IN FILINGS: {insight.is_grounded}")
    
    print(f"\n📑 CITATIONS ({len(insight.citations)} found):")
    for idx, c in enumerate(insight.citations, start=1):
        print(f"  [{idx}] Document: {c.document} | Page: {c.page}")
        print(f"      Quote: \"{c.quote}\"\n")


def main():
    print("Compiling LangGraph Workflow...")
    agent = build_rag_graph()

    # --- TEST 1: Valid Financial & Operational Question ---
    print("\n>>> RUNNING TEST 1: Valid Operational Query <<<")
    state_input_1: AgentState = {
        "query": "What are Nubank's key operational achievements in Mexico and Colombia?",
        "fiscal_filter": None,
        "documents": [],
        "relevant_documents": [],
        "final_insight": None
    }
    
    result_1 = agent.invoke(state_input_1)
    print_result("Test 1 Result", result_1)

    # --- TEST 2: Out-of-Scope / Irrelevant Question ---
    print("\n>>> RUNNING TEST 2: Irrelevant / Out-of-Scope Query <<<")
    state_input_2: AgentState = {
        "query": "What are the specs of the rocket engine used by SpaceX to reach Mars?",
        "fiscal_filter": None,
        "documents": [],
        "relevant_documents": [],
        "final_insight": None
    }
    
    result_2 = agent.invoke(state_input_2)
    print_result("Test 2 Result", result_2)


if __name__ == "__main__":
    main()