# Agentic GraphRAG with LangGraph

This project implements a multi-role Agentic GraphRAG system using LangGraph, DeepSeek, and FAISS, enhanced with a local Knowledge Graph. It features a coordinated "Librarian + Assistant" workflow optimized for AMD GPU acceleration.

## 🌟 Features

-   **Multi-Role Architecture**:
    -   **Librarian Agent**: Handles data ingestion, incremental synchronization, and **Knowledge Graph extraction** using a local LLM.
    -   **Knowledge Assistant**: An agentic RAG assistant that performs **Hybrid Retrieval (Vector + Graph)** with self-correction logic.
-   **Knowledge Graph Integration**: Extracts entities and relationships using a local LLM (Qwen2.5-0.5B) and stores them in a NetworkX-based graph.
-   **Enhanced Entity Linking**: Uses LLM-based entity extraction and fuzzy matching (via `difflib`) to link user queries to graph nodes.
-   **Incremental Ingestion**: Uses MD5 hashing to track file changes. Only new or modified files are processed, making daily updates extremely efficient.
-   **AMD GPU Acceleration**: Leverages **DirectML** via ONNX Runtime for high-performance embedding generation on AMD hardware.
-   **Dynamic Knowledge Reloading**: The Assistant automatically detects and reloads the vector store and graph if the Librarian updates them in the background.

## 🏗️ Architecture

For detailed information about the system design, data flow, and agent coordination, please refer to [ARCHITECTURE.md](ARCHITECTURE.md).

## 🛠️ Tech Stack

-   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
-   **LLM**: DeepSeek (via `langchain-openai`) & Qwen2.5-0.5B (Local)
-   **Embeddings**: [FastEmbed](https://github.com/qdrant/fastembed) (Model: `BAAI/bge-small-en-v1.5`)
-   **Inference Engine**: ONNX Runtime with **DirectML**
-   **Vector Store**: FAISS
-   **Graph Store**: NetworkX
-   **Package Manager**: [uv](https://github.com/astral-sh/uv)

## 🚀 Setup & Usage

Refer to the original sections in this document for [Installation](#setup) and [Execution](#usage) details.

## 🗺️ Roadmap

- [ ] **Scalability**: Transition to Neo4j for professional graph storage and complex queries.
- [ ] **Intelligence**: Multi-hop reasoning and sub-graph retrieval using community detection.
- [ ] **Automation**: Incremental graph updates and schema-defined extraction.
- [ ] **Experience**: Integrated graph visualization UI.

## 📄 License

[MIT](LICENSE)
