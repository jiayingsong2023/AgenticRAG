# Agentic RAG with LangGraph

This project implements an Agentic RAG (Retrieval-Augmented Generation) system using LangGraph, DeepSeek, and FAISS. It features a "Search-Evaluate-Generate" loop with a self-correcting mechanism.

## Features

-   **Agentic Workflow**: Uses LangGraph to orchestrate a stateful retrieval and generation process.
-   **Self-Correction**: Evaluates retrieved documents for relevance. If irrelevant, it rewrites the query and searches again.
-   **Multi-Format Ingestion**: Supports ingestion of PDF, Word (`.docx`), CSV (Email), JSON, and Text (`.txt`) files.
-   **Interactive Mode**: Provides an interactive CLI for chatting with the agent.
-   **Robustness**: Includes a retry limit to prevent infinite loops when no relevant documents are found.

## Tech Stack

-   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
-   **LLM**: DeepSeek (via `langchain-openai`)
-   **Embeddings**: HuggingFace (`sentence-transformers/all-mpnet-base-v2`)
-   **Vector Store**: FAISS
-   **Framework**: LangChain

## Setup

1.  **Install Dependencies**:
    ```bash
    uv sync
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    DEEPSEEK_API_KEY=your_deepseek_api_key
    # Optional: Custom data directory (default is 'data')
    # DATA_DIR=my_documents
    ```

## Usage

### 1. Ingest Data

Place your documents in the `data` directory (or your configured `DATA_DIR`).

Run the ingestion script:
```bash
uv run ingest.py
```
Follow the interactive prompts to confirm the directory and proceed with ingestion. This will create/overwrite the `faiss_index`.

### 2. Run the Agent

Start the interactive agent:
```bash
uv run rag_agent.py
```
Type your question when prompted. The agent will:
1.  Retrieve documents.
2.  Grade them.
3.  Rewrite the query if needed (up to 3 times).
4.  Generate an answer.

Type `exit` or `quit` to stop.

## License

[MIT](LICENSE)