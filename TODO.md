# Agentic Knowledge Graph RAG - Future Roadmap

This document outlines the planned improvements and advanced features for evolving the current POC into a production-grade Agentic GraphRAG system.

## 1. Storage & Scalability
- [ ] **Transition to Neo4j**: Migrate from NetworkX to a professional graph database for persistence, better query performance, and visualization.
- [ ] **Vector-Graph Hybrid Store**: Explore databases that support both vector and graph queries natively (e.g., ArangoDB, NebulaGraph).

## 2. Knowledge Extraction
- [ ] **Advanced LLM for Extraction**: Replace Qwen 0.5B with larger models (Qwen 7B+, DeepSeek-V3, GPT-4o) for higher-quality entity and relation extraction.
- [ ] **Incremental Updates**: Optimize `ingest.py` to only extract graph components from new or modified chunks rather than rebuilding the entire graph.
- [ ] **Schema-Defined Extraction**: Define a fixed ontology/schema for entities and relations to ensure consistency across the graph.

## 3. Retrieval Enhancements
- [ ] **Multi-Hop Reasoning**: Implement nodes in LangGraph that can traverse multiple edges to answer complex relational questions.
- [ ] **Sub-graph Retrieval**: Instead of just 1-hop neighbors, retrieve the most relevant sub-graph cluster using community detection algorithms (e.g., Leiden).
- [ ] **Global Graph Summarization**: Generate high-level summaries of graph communities to answer "big picture" questions (inspired by Microsoft's GraphRAG).
- [ ] **Graph-Enhanced Re-ranking**: Use the graph structure to re-score documents retrieved by the vector store.

## 4. Agent Intelligence
- [ ] **Self-Correction**: Add an agent node to verify if the retrieved graph context actually answers the question, and if not, try a different starting node.
- [ ] **Tool-Use for Graph Queries**: Give the Agent the ability to write Cypher (for Neo4j) or NetworkX queries dynamically.

## 5. UI/UX
- [ ] **Graph Visualization**: Integrate a frontend (like Cytoscape.js or G6) to visualize the retrieved graph context for the user.

