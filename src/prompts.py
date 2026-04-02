from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a router for an enterprise internal document assistant. "
            "Decide whether the question should be answered using the internal knowledge base. "
            "Reply with only YES or NO."
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise knowledge assistant. "
            "Answer using only the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            "Be concise, practical, and include citations inline using the source labels."
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\n"
            "Write a helpful answer. End with a short 'Sources:' section listing the source labels you used."
        ),
    ]
)
