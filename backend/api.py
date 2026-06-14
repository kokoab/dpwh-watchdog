from contextlib import asynccontextmanager

from api_routes.admin import router as admin_router
from api_routes.chat import router as chat_router
from api_routes.embed import router as embed_router
from core.config import cors_allowed_origins
from core.embedding_runtime import clear_embedding_model, load_embedding_model
from memory.chat_memory import initialize_chat_memory_schema
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_chat_memory_schema()
    load_embedding_model()

    yield
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
