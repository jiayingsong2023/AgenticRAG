# Agentic RAG with LangGraph

This project implements an Agentic RAG (Retrieval-Augmented Generation) system using LangGraph, DeepSeek, and FAISS. It features a "Search-Evaluate-Generate" loop with a self-correcting mechanism and is optimized for AMD GPU acceleration.

## Features

-   **Agentic Workflow**: Uses LangGraph to orchestrate a stateful retrieval and generation process.
-   **AMD GPU Acceleration**: Leverages **DirectML** via ONNX Runtime for high-performance embedding generation on AMD hardware (e.g., AI Max+ 395).
-   **Self-Correction**: Evaluates retrieved documents for relevance. If irrelevant, it rewrites the query and searches again.
-   **Multi-Format Ingestion**: Supports ingestion of PDF, Word (`.docx`), CSV, JSON, and Text (`.txt`) files.
-   **Fast Package Management**: Powered by `uv` for lightning-fast dependency management and execution.

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

### 1. Ingest Data

Place your documents in the `data` directory (or your configured `DATA_DIR`).

Run the ingestion script:
```bash
uv run ingest.py
```
The script will automatically detect your AMD GPU and use it for embedding generation. Follow the interactive prompts to confirm the directory.

### 2. Run the Agent

Start the interactive agent:
```bash
uv run rag_agent.py
```
Type your question when prompted. The agent will:
1.  Retrieve documents (using GPU-accelerated embeddings).
2.  Grade them for relevance.
3.  Rewrite the query if needed (up to 3 times).
4.  Generate an answer using DeepSeek LLM.

Type `exit` or `quit` to stop.

## Hardware Optimization

This project is specifically optimized for **AMD AI Max+ 395 (Strix Halo)** and other AMD GPUs. It uses `onnxruntime-directml` to ensure that embedding calculations are offloaded to the GPU, significantly reducing CPU load and improving response times.

## License

[MIT](LICENSE)