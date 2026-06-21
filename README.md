# Enterprise Multimodal RAG Pipeline

An enterprise-grade, 100% local, and containerized Retrieval-Augmented Generation (RAG) pipeline designed for zero-cost ($0) operation, strict data privacy, and production-scale data engineering orchestration. This architecture leverages Semantic Chunking, Hybrid Search, and a Two-Stage Cross-Encoder Reranking pipeline to deliver highly accurate context to a local LLM without external API dependencies or cloud data leakage.

## System Architecture:
![image alt]()


## Key Features:

- PDF ingestion
- Semantic chunking
- Vector embeddings
- Qdrant vector search
- Local LLM inference
- Dockerized deployment

## Tech Stack and Dependencies:
**Service Layer:** FastAPI, Uvicorn

**Vector Infrastructure:** Qdrant DB (Containerized)

**Local LLM Host:** Ollama (llama3 engine)

**Embedding Pipeline:** Sentence-Transformers (all-MiniLM-L6-v2)

**Orchestration & DevOps:** Docker, Docker Compose

**Data Processing:** PyPDF, Pydantic
