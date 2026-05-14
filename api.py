from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from contextlib import asynccontextmanager
from typing import List

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model into RAM and checking for GPU...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ml_models["encoder"] = SentenceTransformer (
        "intfloat/multilingual-e5-small",
        device=device
    )
    
    print(f"Model loaded on: {device}")
    yield
    ml_models.clear()
    print("Model cleared from memory")
    
app = FastAPI(lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    inputs: List[str]
    
@app.post("/embed")
async def embed_text(request: EmbeddingRequest):
    # processed_texts = [f"passage: {t}" for t in request.inputs]
    
    with torch.no_grad():
        embedding = ml_models["encoder"].encode(
            request.inputs,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
    full_embedding = embedding.tolist()

    return {
        "text": request.inputs,
        "embedding": full_embedding,
        "dimensions": len(full_embedding)
    }