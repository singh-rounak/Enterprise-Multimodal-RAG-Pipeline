from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.pipeline import LocalRAGEngine
import os

app = FastAPI(title="Enterprise RAG Service Gateway")
engine = LocalRAGEngine()

class QueryRequest(BaseModel):
    question: str

## CRUD OPERATIONS

# READ
@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "running locally"}

# CREATE
@app.post("/ingest")
def trigger_ingest(file_name: str):
    target_path = f"./data/{file_name}"
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Target document not found in data directory.")
    
    num_chunks = engine.ingest_pdf(target_path)
    return {"status": "success", "file": file_name, "vector_chunks_indexed": num_chunks}

# READ
@app.post("/query")
def execute_query(payload: QueryRequest):
    # 1. Fetch relevant context vectors locally
    context = engine.retrieve_context(query=payload.question)
    
    if not context:
        return {"answer": "No context blocks found. Please ingest documents first.", "context_used": []}
        
    # 2. Run generation via containerized Llama3
    answer = engine.generate_answer(query=payload.question, context=context)
    return {
        "answer": answer,
        "context_sources_parsed": context.split("\n---\n")
    }