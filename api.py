from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI
import asyncio
from concurrent.futures import ThreadPoolExecutor

ml_models = {}
ml_executor = ThreadPoolExecutor(max_workers=4)

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

def encode_texts(texts: list[str]):
    with torch.no_grad():
        return ml_models["encoder"].encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=128
        )

class EmbeddingRequest(BaseModel):
    inputs: List[str]
    
@app.post("/embed")
async def embed_text(request: EmbeddingRequest):
    loop = asyncio.get_running_loop()
    # processed_texts = [f"passage: {t}" for t in request.inputs]
    
    embedding = await loop.run_in_executor(
        ml_executor,
        encode_texts,
        request.inputs
    )
        
    full_embedding = embedding.tolist()

    return {
        "text": request.inputs,
        "embedding": full_embedding,
        "dimensions": len(full_embedding)
    }