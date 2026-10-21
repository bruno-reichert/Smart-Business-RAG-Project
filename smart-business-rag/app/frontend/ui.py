import os
import uuid
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Smart Business Operations RAG",
    page_icon="💼",
    layout="wide"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# --- SESSION STATE INITIALIZATION ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("💼 Ops Intelligence")
    st.caption("Grounded Executive Decision Support")
    st.divider()

    # Session Management
    st.subheader("Conversation Session")
    st.code(st.session_state.thread_id, language="text")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Metadata Filter
    st.subheader("Search Scope")
    quarter_filter = st.selectbox(
        "Fiscal Quarter Filter",
        options=["All Quarters", "2Q25", "3Q25", "4Q25"],
        index=0
    )
    filter_value = None if quarter_filter == "All Quarters" else quarter_filter

    st.divider()

    # Document Upload Section
    st.subheader("Add Quarterly Filings")
    uploaded_file = st.file_uploader("Upload PDF report", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload & Index", use_container_width=True):
            with st.spinner("Processing document and generating embeddings..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"Indexed {data['pages_extracted']} pages ({data['chunks_indexed']} chunks)!")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")


# --- MAIN CHAT INTERFACE ---
st.title("📊 Corporate Operations Copilot")
st.markdown("Ask financial and operational questions backed by **traceable, page-level citations** from quarterly filings.")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander(f"📑 View Citations ({len(message['citations'])})"):
                for idx, c in enumerate(message["citations"], start=1):
                    st.markdown(f"**[{idx}] {c['document']} — Page {c['page']}**")
                    st.caption(f"> \"{c['quote']}\"")

# Chat input prompt
if user_prompt := st.chat_input("Ask an operational or financial question..."):
    # 1. Display user prompt
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing filings with Groq & Gemini..."):
            payload = {
                "query": user_prompt,
                "fiscal_filter": filter_value,
                "thread_id": st.session_state.thread_id
            }
            try:
                response = requests.post(f"{API_BASE_URL}/filings/query", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    insight = result["insight"]
                    answer = insight["answer"]
                    citations = insight.get("citations", [])

                    # Render answer
                    st.markdown(answer)

                    # Render citations dropdown
                    if citations:
                        with st.expander(f"📑 View Citations ({len(citations)})"):
                            for idx, c in enumerate(citations, start=1):
                                st.markdown(f"**[{idx}] {c['document']} — Page {c['page']}**")
                                st.caption(f"> \"{c['quote']}\"")

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server at `{API_BASE_URL}`. Is it running?")