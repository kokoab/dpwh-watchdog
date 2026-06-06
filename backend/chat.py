from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import stream_agent
from query_expand import log_query_expansion, query_expand
import json
import uuid
from typing import Iterator

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message:str
    thread_id: str | None = None
    
def event_stream(message: str, thread_id: str) -> Iterator[str]:
    expanded_message = query_expand(message, thread_id=thread_id)
    log_query_expansion(message, expanded_message, thread_id)

    for event in stream_agent(expanded_message, thread_id):
        yield f"data: {json.dumps(event)}\n\n"

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        event_stream(request.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        }
    )
