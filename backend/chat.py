from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import stream_agent
import json
import uuid
from typing import Iterator

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message:str
    thread_id: str | None = None
    
def event_stream(message: str, thread_id: str) -> Iterator[str]:
    for event in stream_agent(message, thread_id):
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