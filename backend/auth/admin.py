from auth.auth import CurrentUser
from auth.db import require_admin
from chat_memory import list_chat_messages, list_chat_threads
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/chat-threads")
async def admin_list_chat_threads(
    limit: int = 100,
    current_user: CurrentUser = Depends(require_admin),
):
    return {
        "threads": list_chat_threads(
            user_id=None,
            limit=max(1, min(limit, 500)),
        )
    }


@router.get("/chat-threads/{thread_id}/messages")
async def admin_get_chat_messages(
    thread_id: str,
    limit: int = 500,
    current_user: CurrentUser = Depends(require_admin),
):
    return {
        "thread_id": thread_id,
        "messages": list_chat_messages(
            thread_id,
            user_id=None,
            limit=max(1, min(limit, 1000)),
        ),
    }
