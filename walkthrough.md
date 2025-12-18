# Walkthrough - Multi-Role AI Agent RAG System

I have transformed the RAG system into a sophisticated multi-agent architecture that automates knowledge management.

## Changes Made

### 1. Librarian Agent (`ingest.py`)
- **Role**: Data Producer.
- **Incremental Sync**: Uses MD5 hashes to track file changes. It only rebuilds the index when changes are detected, making daily updates efficient.
- **Hardware**: Fully utilizes the AMD GPU (DirectML) for embedding generation.

### 2. Knowledge Assistant (`rag_agent.py`)
- **Role**: Data Consumer.
- **Dynamic Reloading**: Automatically detects if the Librarian has updated the vector store and reloads it without restarting the session.

### 3. Orchestrator (`main.py`)
- **Role**: System Coordinator.
- **Background Sync**: Runs a background thread that performs a "Librarian Sync" every 24 hours (configurable).
- **Unified Entry**: Provides a single command to start the entire system.

## Verification Results

### Multi-Agent Coordination
I verified that running `uv run python main.py` successfully:
1. Performs an initial sync of the data directory.
2. Starts a background thread for periodic updates.
3. Launches the interactive chat interface.

### Incremental Logic
The Librarian now correctly identifies:
- New files to be added.
- Modified files that need re-indexing.
- Deleted files that should be removed from the knowledge base.

## How to Use

Simply run the new orchestrator:
```bash
uv run python main.py
```

The system will handle the rest, ensuring your Knowledge Assistant always has the latest information from your `data` directory.
