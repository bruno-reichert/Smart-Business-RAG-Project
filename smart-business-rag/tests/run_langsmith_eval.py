import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

# Safe API key check
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("Neither LANGSMITH_API_KEY nor LANGCHAIN_API_KEY is set in .env!")
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

from langsmith import evaluate
from app.services.graph_builder import build_rag_graph
from app.services.agent_state import AgentState

agent = build_rag_graph()

def target_pipeline(inputs: dict) -> dict:
    """The function LangSmith calls for each example in the dataset."""
    query = inputs["query"]
    state_input: AgentState = {
        "query": query,
        "fiscal_filter": None,
        "documents": [],
        "relevant_documents": [],
        "final_insight": None
    }
    
    result = agent.invoke(state_input)
    insight = result.get("final_insight")
    
    return {
        "answer": insight.answer if insight else "No answer",
        "is_grounded": insight.is_grounded if insight else False,
        "citations_count": len(insight.citations) if insight else 0
    }

def grounding_evaluator(run, example) -> dict:
    """Evaluates whether the system marked the output as grounded with citations."""
    is_grounded = run.outputs.get("is_grounded", False)
    citations_count = run.outputs.get("citations_count", 0)
    score = 1.0 if (is_grounded and citations_count > 0) else 0.0
    return {"key": "grounded_with_citations", "score": score}

def main():
    print("🚀 Triggering LangSmith Cloud Evaluation Experiment...")
    
    experiment_results = evaluate(
        target_pipeline,
        data="nubank-operational-qa",
        evaluators=[grounding_evaluator],
        experiment_prefix="agentic-rag-benchmark",
        max_concurrency=1  # Sequential execution to be gentle on free-tier rate limits
    )
    
    print("\n✅ LangSmith Experiment Completed!")
    print("Check your dashboard: https://smith.langchain.com -> Datasets & Experiments -> 'Experiments' tab!")

if __name__ == "__main__":
    main()