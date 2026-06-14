from typing import List

from core.embedding_runtime import embed_inputs
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class EmbeddingRequest(BaseModel):
    inputs: List[str]


@router.post("/embed")
async def embed_text(request: EmbeddingRequest):
    return await embed_inputs(request.inputs)
