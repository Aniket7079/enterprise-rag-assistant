from typing import TypedDict, List, Dict, Any
from unittest import result

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from sqlalchemy import text

from .llm import get_llm
from .prompts import ROUTER_PROMPT, ANSWER_PROMPT
from .retriever import get_retriever


class RAGState(TypedDict, total=False):
    question: str
    route: str
    context: str
    sources: List[str]
    answer: str
    retrieved_docs: List[Dict[str, Any]]


def _docs_to_context(docs: List[Document]) -> tuple[str, List[str], List[Dict[str, Any]]]:
    lines = []
    sources = []
    structured_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "na")
        label = f"{source}#chunk-{chunk_id}"
        sources.append(label)
        structured_docs.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "content": doc.page_content,
            }
        )
        lines.append(f"[{label}]\n{doc.page_content}")

    dedup_sources = list(dict.fromkeys(sources))
    return "\n\n---\n\n".join(lines), dedup_sources, structured_docs


def classify_query(state: RAGState) -> RAGState:
    llm = get_llm()
    prompt = ROUTER_PROMPT.format_messages(question=state["question"])
    result = llm.invoke(prompt)
# handle both str and AIMessage
    text = result if isinstance(result, str) else result.content
    print("FINAL ANSWER:", text)   # debug
    return {"answer": text}
    text = result if isinstance(result, str) else result.content
    response = text.strip().upper()
    route = "retrieve" if response.startswith("YES") else "direct"
    return {"route": route}


def retrieve_context(state: RAGState) -> RAGState:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    context, sources, structured_docs = _docs_to_context(docs)
    return {"context": context, "sources": sources, "retrieved_docs": structured_docs}


def direct_answer(state: RAGState) -> RAGState:
    llm = get_llm()
    prompt = (
        "You are a helpful assistant. This question does not require internal documents. "
        "Answer briefly and directly.\n\n"
        f"Question: {state['question']}"
    )
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer, "sources": []}


def generate_answer(state: RAGState) -> RAGState:
    llm = get_llm()
    prompt = ANSWER_PROMPT.format_messages(
        question=state["question"],
        context=state.get("context", ""),
    )
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer, "sources": state.get("sources", [])}


def build_rag_app():
    builder = StateGraph(RAGState)

    builder.add_node("classify", classify_query)
    builder.add_node("retrieve", retrieve_context)
    builder.add_node("direct", direct_answer)
    builder.add_node("generate", generate_answer)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "direct": "direct",
        },
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("direct", END)
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
