import os
import json
from typing import Dict, Any, List, cast
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.agent_state import AgentState
from app.services.vector_store import similarity_search_with_metadata
from app.models.schemas import OperationalInsight, Citation

load_dotenv()

# 1. Initialize our Dual-Engine LLMs
# Groq: For ultra-fast micro-evaluations
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0
)

# Gemini: For high-accuracy synthesis and structured output
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.0
)

# --- NODE 0: CONTEXTUAL QUERY REFORMULATOR (Groq) ---
def reformulate_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes recent conversation history. If the user's latest query is a follow-up
    or contains ambiguous pronouns (e.g., 'what about Colombia?'), Groq rewrites it
    into a self-contained search query before vector retrieval.
    """
    messages = state.get("messages", [])
    raw_query = state["query"]
    
    # If this is the very first turn (no prior messages), skip reformulation
    if len(messages) <= 1:
        print(f"[Node: Query Reformulator] First turn detected. Using raw query: '{raw_query}'")
        return {
            "query": raw_query,
            "messages": [HumanMessage(content=raw_query)]
        }

    print(f"[Node: Query Reformulator] Context detected ({len(messages)} prior messages). Analyzing with Groq...")

    # Format recent history for the rewriter (last 4 messages maximum to keep prompt lean)
    recent_history = messages[-4:]
    history_str = ""
    for msg in recent_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content_preview = str(msg.content)[:200]
        history_str += f"{role}: {content_preview}\n"

    system_prompt = (
        "You are an expert query reformulation assistant for a financial research system.\n"
        "Given the recent conversation history and a follow-up user query, rewrite the user query "
        "into a standalone, unambiguous search query that can be used for semantic document retrieval.\n"
        "CRITICAL RULES:\n"
        "1. Do NOT answer the question. Only rewrite it.\n"
        "2. If the user query is ALREADY standalone and unambiguous, return it EXACTLY as is.\n"
        "3. Output ONLY the rewritten search query, with no quotation marks, preambles, or explanations."
    )

    user_prompt = (
        f"Conversation History:\n{history_str}\n"
        f"Latest User Query: {raw_query}\n\n"
        f"Standalone Search Query:"
    )

    try:
        response = groq_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        rewritten = str(response.content).strip().strip('"').strip("'")
        print(f"[Node: Query Reformulator] Reformulated: '{raw_query}' -> '{rewritten}'")
        return {
            "query": rewritten,
            "messages": [HumanMessage(content=raw_query)]
        }
    except Exception as e:
        print(f"Warning: Query reformulation failed, falling back to raw query. Error: {e}")
        return {
            "query": raw_query,
            "messages": [HumanMessage(content=raw_query)]
        }

# --- NODE 1: RETRIEVE ---
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves candidate chunks from Supabase pgvector based on query & optional filter."""
    query = state["query"]
    fiscal_filter = state.get("fiscal_filter")
    
    filter_dict = {"fiscal_period": fiscal_filter} if fiscal_filter else None
    
    print(f"\n[Node: Retrieve] Searching Supabase for query: '{query}' (Filter: {filter_dict})")
    retrieved_docs = similarity_search_with_metadata(query=query, k=4, filter_dict=filter_dict)
    print(f"[Node: Retrieve] Found {len(retrieved_docs)} candidate chunks.")
    
    return {"documents": retrieved_docs}


# --- NODE 2: GRADE DOCUMENTS (Groq) ---
def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates retrieved chunks for relevance using Groq's high-speed inference.
    Filters out background noise to prevent hallucinated answers.
    """
    query = state["query"]
    documents = state["documents"]
    
    print(f"[Node: Grade Documents] Evaluating {len(documents)} chunks with Groq...")
    
    system_prompt = (
        "You are an expert financial analyst assistant. Your task is to evaluate if a retrieved document "
        "chunk contains information relevant to the user query.\n"
        "Respond strictly with a JSON object with a single boolean field: {\"is_relevant\": true} or {\"is_relevant\": false}."
    )
    
    relevant_docs: List[Document] = []
    
    for doc in documents:
        user_prompt = f"User Query: {query}\n\nDocument Snippet:\n{doc.page_content}"
        
        try:
            response = groq_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Simple cleanup to parse json response safely
            raw_content = response.content
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            content = content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            
            parsed = json.loads(content)
            if parsed.get("is_relevant", False):
                relevant_docs.append(doc)
        except Exception as e:
            # Fallback: keep document if grading fails
            print(f"Warning: Grading failed for chunk, keeping it. Error: {e}")
            relevant_docs.append(doc)
            
    print(f"[Node: Grade Documents] {len(relevant_docs)} / {len(documents)} chunks determined relevant.")
    return {"relevant_documents": relevant_docs}


# --- NODE 3: GENERATE (Gemini) ---
def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes the final executive insight using Gemini with structured Pydantic outputs.
    Ensures every fact has a direct, page-level citation.
    """
    query = state["query"]
    relevant_docs = state["relevant_documents"]
    
    print(f"[Node: Generate] Synthesizing executive answer with Gemini from {len(relevant_docs)} chunks...")
    
    # Format the context blocks with clear document & page provenance
    context_blocks = []
    for doc in relevant_docs:
        meta = doc.metadata
        context_blocks.append(
            f"--- SOURCE: {meta.get('source', 'Unknown')} | PAGE: {meta.get('page', 0)} | QUARTER: {meta.get('fiscal_period', '')} ---\n"
            f"{doc.page_content}\n"
        )
    formatted_context = "\n".join(context_blocks)
    
    prompt = (
        f"You are a Senior Corporate Intelligence Analyst at Nubank.\n"
        f"Answer the user query based ONLY on the following filing excerpts.\n"
        f"Provide exact page-level citations for every single metric and claim mentioned.\n\n"
        f"Context Excerpts:\n{formatted_context}\n\n"
        f"User Query: {query}"
    )
    
    # We bind the Pydantic schema directly to Gemini for deterministic structured output
    structured_gemini = gemini_llm.with_structured_output(OperationalInsight)
    
    raw_insight = structured_gemini.invoke(prompt)
    insight: OperationalInsight = cast(OperationalInsight, raw_insight)
    print("[Node: Generate] Answer synthesized successfully with citations.")
    
    return {
            "final_insight": insight,
            "messages": [AIMessage(content=insight.answer)]
        }

# --- NODE 4: OUT OF BOUNDS / FALLBACK ---
def fallback_node(state: AgentState) -> Dict[str, Any]:
    """Gracefully handles queries when no relevant context was found in the filings."""
    query = state["query"]
    print("[Node: Fallback] No relevant documents found. Triggering ungrounded safety fallback.")
    
    fallback_insight = OperationalInsight(
        answer=(
            f"Based on the processed quarterly filings (3Q25 and 4Q25), there is no verifiable "
            f"information available to answer: '{query}'. Please verify if the requested metric "
            f"pertains to a different reporting period or company."
        ),
        citations=[],
        is_grounded=False
    )
    return {
        "final_insight": fallback_insight,
        "messages": [AIMessage(content=fallback_insight.answer)]
    }