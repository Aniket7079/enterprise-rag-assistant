import streamlit as st
from dotenv import load_dotenv

from src.config import DEFAULT_THREAD_ID
from src.graph import build_rag_app
from src.loader import load_documents 
from src.retriever import ensure_vectorstore


load_dotenv()

st.set_page_config(
    page_title="Enterprise Internal Document Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("Enterprise Internal Document Assistant")
st.caption("A mid-level LangChain + LangGraph RAG project with Chroma, embeddings, and source-grounded answers.")

with st.sidebar:
    st.header("Controls")
    thread_id = st.text_input("Conversation thread ID", value=DEFAULT_THREAD_ID)
    refresh = st.button("Build / refresh index")
    st.write("Documents found:", len(load_documents()))
    st.write("Thread ID:", thread_id)

if refresh:
    with st.spinner("Indexing documents..."):
        ensure_vectorstore(force_rebuild=True)
    st.success("Index refreshed.")

graph_app = build_rag_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the internal knowledge base")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph_app.invoke(
                {"question": question},
                config={"configurable": {"thread_id": thread_id}},
            )

        answer = result["answer"]
        sources = result.get("sources", [])

        st.markdown(answer)
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                st.code(source)

    st.session_state.messages.append({"role": "assistant", "content": answer})
