from contextlib import asynccontextmanager

from core.config import cors_allowed_origins
from core.database import close_pool, init_pool
from core.embedding_runtime import clear_embedding_model, load_embedding_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.admin.router import router as admin_router
from features.chat.memory import initialize_chat_memory_schema
from features.chat.router import router as chat_router
from features.embed.router import router as embed_router
from features.library.router import router as library_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_chat_memory_schema()
    init_pool()
    load_embedding_model()

    yield
    close_pool()
    clear_embedding_model()


app = FastAPI(lifespan=lifespan)

allowed_origins = cors_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(embed_router)
app.include_router(library_router)
