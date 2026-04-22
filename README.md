# Enterprise Internal Document Assistant
  
A mid-level end-to-end RAG project for an internal enterprise knowledge assistant.

## What this project shows
             
- **RAG pipeline**: document loading → chunking → embeddings → vector search → LLM answer generation
- **LangGraph orchestration**: query routing, retrieval node, answer node, and optional direct-answer branch
- **LangChain integrations**: chat model, retriever, prompt templates, and document abstractions
- **Vector database**: Chroma persistence for local semantic search
- **Resume value**: realistic enterprise use case with citations and a clean, demo-friendly UI

## Recommended stack

- Python
- LangChain
- LangGraph
- Chroma
- Sentence Transformers embeddings
- OpenAI or Ollama as the LLM provider
- Streamlit for the UI

## Project strucTURE

```text
enterprise-rag-assistant/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── data/
│   └── docs/
├── chroma_db/
└── src/
    ├── config.py
    ├── embeddings.py
    ├── graph.py
    ├── llm.py
    ├── loader.py
    ├── prompts.py
    └── retriever.py
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in the values you need.

### Option A: OpenAI LLM
Set `OPENAI_API_KEY` and use the default provider.

### Option B: Local Ollama LLM
Install Ollama locally and set:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Build the vector index

```bash
python ingest.py
```

This reads documents from `data/docs/`, chunks them, embeds them, and stores them in `chroma_db/`.

## Run the app

```bash
streamlit run app.py
```

## Example questions

- What is the leave approval process?
- How do I submit an expense claim?
- What should I do if my laptop is not working?
- What documents are needed during onboarding?

## Why this is a strong resume project

This project is practical because it resembles a real company knowledge assistant. It demonstrates:
- semantic retrieval instead of keyword search,
- LLM reasoning over grounded context,
- graph-based orchestration,
- source citations,
- and a production-style local persistence layer for vectors.

## Notes

- The sample documents are generic and intentionally small.
- You can replace them with PDFs, DOCX files, or internal wiki exports later.
- For production, switch from in-memory LangGraph checkpointing to a durable store such as SQLite or Postgres.
