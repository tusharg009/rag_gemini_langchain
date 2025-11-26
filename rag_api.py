import os
import uuid
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gemini_call import process_gemini_query


DATA_FOLDER = "./data"
MODEL_NAME = "nomic-embed-text"   # externalized embedding model

app = FastAPI(title="RAG API - Ollama + Chroma")

# Global embedding model
embedder = OllamaEmbeddings(model=MODEL_NAME)

# Chroma client + collection
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# Utility Functions
# -----------------------------
def read_file(filename: str) -> str:
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def upsert_document(filename: str):
    raw_text = read_file(filename)
    chunks = chunk_text(raw_text)

    document_id = str(uuid.uuid4())

    # Remove existing chunks for filename
    existing = collection.get(where={"filename": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Embed
    embeddings = embedder.embed_documents(chunks)

    # Add new chunks
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metas = [
        {"filename": filename, "doc_id": document_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metas
    )

    return {"document_id": document_id, "chunks": len(chunks)}


def retrieve(query: str, k=1):
    query_vec = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k
    )
    return results


# -----------------------------
# Request Models
# -----------------------------
class UpsertRequest(BaseModel):
    filename: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 1


# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/upsert")
def upsert_api(req: UpsertRequest):
    try:
        result = upsert_document(req.filename)
        return {"status": "success", **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query")
def query_api(req: QueryRequest):
    print("query receive: ", req.query)
    results = retrieve(req.query, req.top_k)

    # call LLM here
    obj_to_pass = {"status": "success", "results": results, "query": req.query}
    llm_message = process_gemini_query(obj_to_pass);
    return {"llm_message": llm_message}
