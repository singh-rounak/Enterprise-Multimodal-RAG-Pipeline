import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests

class LocalRAGEngine:
    def __init__(self):
        # Connect to containerized Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.client = QdrantClient(host=qdrant_host, port=6333)
        
        # Load local embedding model (Free, runs on your CPU)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "enterprise_knowledge"
        self.ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize collection if not existing
        self._init_vector_db()

    def _init_vector_db(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384, # Dimensionality of all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )

    def ingest_pdf(self, file_path: str):
        """Extracts text, splits into semantic chunks, embeds, and uploads to Qdrant."""
        reader = PdfReader(file_path)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""
            
        # Recursive Chunking to protect contextual integrity
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)
        
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.encoder.encode(chunk).tolist()
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "chunk_id": idx
                    }
                )
            )
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(chunks)

    def retrieve_context(self, query: str, limit: int = 3) -> str:
        """Performs vector search and compiles relevant contextual strings."""
        query_vector = self.encoder.encode(query).tolist()
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        context_blocks = [result.payload["text"] for result in search_results]
        return "\n---\n".join(context_blocks)

    def generate_answer(self, query: str, context: str) -> str:
        """Routes payload to the local Ollama LLM wrapper container."""
        prompt = f"""You are a secure, enterprise assistant. Answer the question using ONLY the provided context. If the context does not contain the answer, say "Information not found in system logs."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        ANSWER:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=30
            )
            return response.json().get("response", "Error generating response.")
        except Exception as e:
            return f"Failed to connect to local LLM Engine: {str(e)}"