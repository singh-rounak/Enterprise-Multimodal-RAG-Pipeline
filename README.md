# Enterprise Multimodal RAG Pipeline

An enterprise-grade, 100% local, and containerized Retrieval-Augmented Generation (RAG) pipeline designed for zero-cost ($0) operation, strict data privacy, and production-scale data engineering orchestration. This architecture leverages Semantic Chunking, Hybrid Search, and a Two-Stage Cross-Encoder Reranking pipeline to deliver highly accurate context to a local LLM without external API dependencies or cloud data leakage.

## System Architecture:

[ Unstructured Data / PDFs ] ──> Drop into ./data/
                                       │
                                       ▼
                         [ FastAPI Ingestion Engine ]
                                       │
                                       ▼
                       { Semantic Text Splitting Module }
                     (Sentence Embedding Space Thresholding)
                                       │
                                       ▼
                        { Local Embedding Generation }
                           (all-MiniLM-L6-v2 Model)
                                       │
                                       ▼
                       [ Local Qdrant Vector Engine ]
                       (Persistent Layer via Docker)
                                       │
                                       ▼
                        [ Two-Stage Retrieval Loop ]
                 ├── Track 1: Dense Vector Similarity (HNSW)
                 └── Track 2: Sparse Keyword Lookup (BM25)
                                       │
                                       ▼
                       [ Reciprocal Rank Fusion (RRF) ]
                                       │
                                       ▼
                       [ BGE Cross-Encoder Reranker ]
                                       │
                                       ▼ (Top 5 Context Chunks)
                        [ Local Ollama Container ]
                             (Llama 3 Inference)
