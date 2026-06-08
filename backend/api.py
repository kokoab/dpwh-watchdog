import asyncio
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from chat_memory import initialize_chat_memory_schema
from chat import router as chat_router

ml_models = {}
BATCH_WAIT_MS = 30
DYNAMIC_BATCH_MAX = 256
MAX_BATCH_TEXTS = 256
ENCODE_BATCH_SIZE = 256

request_queue: asyncio.Queue | None = None


async def batch_worker():
    if request_queue is None:
        raise HTTPException(status_code=503, detail="Request queue not available")

    loop = asyncio.get_event_loop()
    pending_item = None

    while True:
        if pending_item is None:
            first = await request_queue.get()
        else:
            first = pending_item
            pending_item = None
        batch = [first]
        batch_texts = len(first[0])

        try:
            deadline = loop.time() + (BATCH_WAIT_MS / 1000)
            while len(batch) < DYNAMIC_BATCH_MAX:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                item = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                item_texts = len(item[0])
                if batch_texts + item_texts > MAX_BATCH_TEXTS and batch:
                    pending_item = item
                    break
                batch.append(item)
                batch_texts += item_texts
        except asyncio.TimeoutError:
            pass

        all_texts = [text for texts, _ in batch for text in texts]

        try:
            vectors = await loop.run_in_executor(None, encode_texts, all_texts)
        except Exception as e:
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
            continue

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        offset = 0
        for texts, future in batch:
            n = len(texts)
            if not future.done():
                future.set_result(vectors[offset : offset + n])
            offset += n


@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue, MAX_BATCH_TEXTS, ENCODE_BATCH_SIZE
    print("Loading model into RAM and checking for GPU...")
    initialize_chat_memory_schema()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # device = "cpu"

    ml_models["encoder"] = SentenceTransformer(
        "intfloat/multilingual-e5-small", device=device
    )

    # BAAI/bge-m3

    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.8)
        MAX_BATCH_TEXTS = 256
        ENCODE_BATCH_SIZE = 128
    else:
        MAX_BATCH_TEXTS = 1024
        ENCODE_BATCH_SIZE = 256

    print(f"Model loaded on: {device}")

    request_queue = asyncio.Queue()

    worker = asyncio.create_task(batch_worker())

    yield
    worker.cancel()
    ml_models.clear()
    print("Model cleared from memory")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)


def encode_texts(texts: list[str]):
    with torch.no_grad():
        result = ml_models["encoder"].encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=ENCODE_BATCH_SIZE,
        )
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


class EmbeddingRequest(BaseModel):
    inputs: List[str]


@app.post("/embed")
async def embed_text(request: EmbeddingRequest):
    if request_queue is None:
        raise HTTPException(status_code=503, detail="Request queue not available")

    if not request.inputs:
        raise HTTPException(status_code=400, detail="inputs list is empty")
    if len(request.inputs) > DYNAMIC_BATCH_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inputs: {len(request.inputs)} (max {DYNAMIC_BATCH_MAX})",
        )

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await request_queue.put((request.inputs, future))

    # Wait for batch_worker to complete this request's slice
    vectors = await future

    return {
        "text": request.inputs,
        "embedding": vectors.tolist(),
        "dimensions": vectors.shape[1],
    }
