import asyncio

import torch
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

ml_models = {}
BATCH_WAIT_MS = 30
DYNAMIC_BATCH_MAX = 256
MAX_BATCH_TEXTS = 256
ENCODE_BATCH_SIZE = 256

request_queue: asyncio.Queue | None = None
worker_task: asyncio.Task | None = None


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


def load_embedding_model():
    global request_queue, worker_task, MAX_BATCH_TEXTS, ENCODE_BATCH_SIZE

    print("Loading model into RAM and checking for GPU...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ml_models["encoder"] = SentenceTransformer(
        "intfloat/multilingual-e5-small", device=device
    )

    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.8)
        MAX_BATCH_TEXTS = 256
        ENCODE_BATCH_SIZE = 128
    else:
        MAX_BATCH_TEXTS = 1024
        ENCODE_BATCH_SIZE = 256

    print(f"Model loaded on: {device}")

    request_queue = asyncio.Queue()
    worker_task = asyncio.create_task(batch_worker())


def clear_embedding_model():
    global worker_task

    if worker_task is not None:
        worker_task.cancel()
        worker_task = None
    ml_models.clear()
    print("Model cleared from memory")


async def embed_inputs(inputs: list[str]):
    if request_queue is None:
        raise HTTPException(status_code=503, detail="Request queue not available")

    if not inputs:
        raise HTTPException(status_code=400, detail="inputs list is empty")
    if len(inputs) > DYNAMIC_BATCH_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inputs: {len(inputs)} (max {DYNAMIC_BATCH_MAX})",
        )

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await request_queue.put((inputs, future))
    vectors = await future

    return {
        "text": inputs,
        "embedding": vectors.tolist(),
        "dimensions": vectors.shape[1],
    }
