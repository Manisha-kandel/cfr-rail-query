"""Streamlit chat UI for the CFR Rail Query RAG system."""

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(
    page_title="CFR Rail Query",
    page_icon="🚂",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

MAX_HISTORY = 5  # keep last 5 exchanges

st.info(
    "📚 Knowledge Base: 49 CFR Parts 213, 214, 225, 229, 232, 234 — "
    "as of June 2026 | ⚠️ For reference only. Always verify against "
    "official current CFR."
)

with st.sidebar:
    st.title("About")
    st.write(
        "CFR Rail Query is a regulatory reference assistant for "
        "railroad professionals."
    )
    st.write(
        "Ask questions about track safety, roadway worker protection, "
        "locomotive standards, brake systems, accident reporting, and "
        "grade crossing signals."
    )
    st.divider()
    st.write("Powered by:")
    st.write("• 49 CFR Title 49 Standards")
    st.write("• LangChain + FAISS")
    st.write("• GPT-4o-mini")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render a collapsed expander listing source citations."""
    if not sources:
        return
    # Deduplicate by (file, page) pair
    seen = set()
    unique_sources = []
    for source in sources:
        key = (source['file'], source['page'])
        if key not in seen:
            seen.add(key)
            unique_sources.append(source)
    with st.expander("📄 Sources (click to expand)"):
        for source in unique_sources:
            st.write(f"• {source['file']} — page {source['page']}")


def render_table_references(table_references: list[dict[str, Any]]) -> None:
    """Render a highlighted box listing referenced regulatory tables."""
    if not table_references:
        return
    lines = ["📊 **Regulatory Tables Referenced**", ""]
    for ref in table_references:
        lines.append(
            f"• 49 CFR §{ref['section']} — {ref['cfr_part_title']}\n"
            f"  📄 See: {ref['file']}, page {ref['page']}"
        )
    lines.append("")
    lines.append(
        "ℹ️ Table values should be verified directly in the source document."
    )
    st.info("\n\n".join(lines))


def render_message(message: dict[str, Any]) -> None:
    """Render a single chat message, including sources/table references
    for assistant turns."""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_table_references(message.get("table_references", []))


for message in st.session_state["messages"]:
    render_message(message)

user_input = st.chat_input("Ask a railroad safety regulation question...")

if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state["messages"].append(user_message)
    render_message(user_message)

    with st.spinner("Looking up regulations..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "question": user_input,
                    "chat_history": st.session_state["chat_history"]
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            st.error(
                "Could not reach the API. Make sure the FastAPI server "
                "is running on port 8000."
            )
        else:
            if "answer" not in data:
                st.error("Unexpected response format.")
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "table_references": data.get("table_references", []),
                }
                st.session_state["messages"].append(assistant_message)
                render_message(assistant_message)

                # Add to rolling chat history (max 5 exchanges)
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant",
                     "content": data["answer"]}
                )
                # Keep only last MAX_HISTORY exchanges (each
                # exchange = 1 user + 1 assistant = 2 messages)
                max_messages = MAX_HISTORY * 2
                if len(st.session_state["chat_history"]) > max_messages:
                    st.session_state["chat_history"] = (
                        st.session_state["chat_history"][-max_messages:]
                    )
