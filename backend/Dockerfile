FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# 1. Install CPU-only PyTorch first using the correct index URL
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# 2. Install your remaining dependencies
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
    sentence-transformers

COPY . .

EXPOSE 8000

