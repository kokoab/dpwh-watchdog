FROM python:3.11-slim

WORKDIR /app

# Combine system updates and clean up to keep image size small
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (removed duplicate sentence-transformers)
RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-ollama \
    langchain-huggingface \
    langchain-text-splitters \
    langchain-chroma \
    langchainhub \
    langgraph \
    ddgs \
    bs4 \
    fastapi \
    uvicorn \
    python-dotenv \
    ollama \
    sentence-transformers \
    pgvector psycopg2-binary

# Explicitly set cache dir and download model weights during build
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Copy application files last to optimize layer caching
COPY . .

EXPOSE 8000

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
