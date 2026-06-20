import requests

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(
    host="qdrant",
    port=6333
)   

COLLECTION_NAME = "documents"

def retrieve(query, top_k=5):
    embedding = model.encode(query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k
    )
    context = "\n".join([hit.payload["text"] for hit in search_result])
    return context

def ask_llama(question, context):
    prompt = f"""
    Use the provided context to answer the question. If the context does not contain the answer, say "I don't know".

    Context: {context}
    
    Question: {question}
"""
    
    response = requests.post(
        "http://ollama:11434/api/generate",
        json={"model" : "llama3",
              "prompt": prompt,
              "stream": False,
              "max_length": 200}
    )
    return response.json()["response"]

def answer(question):
    context = retrieve(question)
    answer = ask_llama(question, context)
    
    return answer

