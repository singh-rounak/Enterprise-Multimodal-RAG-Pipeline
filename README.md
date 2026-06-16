# Enterprise Multimodal RAG Pipeline

An enterprise-grade, 100% local, and containerized Retrieval-Augmented Generation (RAG) pipeline designed for zero-cost ($0) operation, strict data privacy, and production-scale data engineering orchestration. This architecture leverages Semantic Chunking, Hybrid Search, and a Two-Stage Cross-Encoder Reranking pipeline to deliver highly accurate context to a local LLM without external API dependencies or cloud data leakage.

## System Architecture:
See image*


## Key Features
1. Adaptive Semantic Chunking
Instead of splitting text using arbitrary character counts which break sentences in half, this system evaluates the semantic distance between consecutive sentences. A chunk boundary is dynamically injected only when the cosine distance exceeds the 95th percentile threshold of the document context, maintaining strict semantic boundaries.

2. High-Performance Vector Storage
Utilizes a containerized Qdrant instance running Hierarchical Navigable Small World (HNSW) graphs. Vectors are stored alongside production-grade metadata payloads (source_file, chunk_id, timestamp) allowing for granular data lineage and partitioned lookups.

3. Two-Stage Hybrid Retrieval & Reranking
To prevent the common "Lost in the Middle" phenomenon found in LLMs, retrieval operates on a dual-track architecture:

Dense Search: Identifies abstract concepts using Cosine Similarity.

Sparse Search: Evaluates strict domain-specific keywords and identifiers using the BM25 algorithm.

Results are combined via Reciprocal Rank Fusion (RRF)

The top 25 merged results are filtered down to the top 5 highly specific chunks using the BAAI/bge-reranker-large Cross-Encoder model, cutting LLM context noise and optimizing processing speed.

## Tech Stack and Dependencies:
**Service Layer:** FastAPI, Uvicorn

**Vector Infrastructure:** Qdrant DB (Containerized)

**Local LLM Host:** Ollama (llama3 engine)

**Embedding Pipeline:** Sentence-Transformers (all-MiniLM-L6-v2)

**Orchestration & DevOps:** Docker, Docker Compose

**Data Processing:** PyPDF, LangChain Text Splitters
