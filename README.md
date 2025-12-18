# Agentic RAG with LangGraph

This project implements a multi-role Agentic RAG system using LangGraph, DeepSeek, and FAISS. It features a coordinated "Librarian + Assistant" workflow optimized for AMD GPU acceleration.

## Features

-   **Multi-Role Architecture**:
    -   **Librarian Agent**: Handles data ingestion, incremental synchronization, and knowledge base maintenance.
    -   **Knowledge Assistant**: An agentic RAG assistant that handles user queries with self-correction logic.
-   **Incremental Ingestion**: Uses MD5 hashing to track file changes. Only new or modified files are processed, making daily updates extremely efficient.
-   **AMD GPU Acceleration**: Leverages **DirectML** via ONNX Runtime for high-performance embedding generation on AMD hardware (e.g., AI Max+ 395).
-   **Dynamic Knowledge Reloading**: The Assistant automatically detects and reloads the vector store if the Librarian updates it in the background.
-   **Self-Correction Loop**: Evaluates retrieved documents for relevance and automatically rewrites queries if needed.
-   **Fast Package Management**: Powered by `uv` for lightning-fast dependency management.

## Tech Stack

-   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
-   **LLM**: DeepSeek (via `langchain-openai`)
-   **Embeddings**: [FastEmbed](https://github.com/qdrant/fastembed) (Model: `BAAI/bge-small-en-v1.5`)
-   **Inference Engine**: ONNX Runtime with **DirectML**
-   **Vector Store**: FAISS
-   **Package Manager**: [uv](https://github.com/astral-sh/uv)

## Setup

1.  **Install Dependencies**:
    Make sure you have `uv` installed. Then run:
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

### 🚀 The Unified Entry Point (Recommended)

Start the entire system (Librarian + Assistant) with a single command:
```bash
uv run python main.py
```
This will:
1.  Perform an initial knowledge sync.
2.  Start the **Librarian** in the background (checks for updates every 24 hours).
3.  Launch the interactive **Knowledge Assistant**.

### 📚 Manual Ingestion
If you only want to update the knowledge base without starting the assistant:
```bash
uv run python ingest.py
```

### 💬 Standalone Assistant
If you want to run the assistant without the background sync:
```bash
uv run python rag_agent.py
```

## Hardware Optimization

This project is specifically optimized for **AMD AI Max+ 395 (Strix Halo)** and other AMD GPUs. It uses `onnxruntime-directml` to ensure that embedding calculations are offloaded to the GPU, significantly reducing CPU load and improving response times.

## License

[MIT](LICENSE)