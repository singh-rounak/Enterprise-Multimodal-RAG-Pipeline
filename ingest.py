import fitz
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

model = SentenceTransformer('all-MiniLM-L6-v2')

client= QdrantClient(
    host="qdrant",
    port=6333
)

COLLECTION_NAME = "documents"

def create_collection():
    collections = [ c.name for c in client.get_collections().collections ]

    if COLLECTION_NAME not in collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def ingest_pdf(filepath):
    create_collection()
    doc = fitz.open(filepath)
    for page in doc:
        text = page.get_text()
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            point = PointStruct(
                id=None,
                vector=embedding,
                payload={"text": chunk}
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
    return len(chunks)


