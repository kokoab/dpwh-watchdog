import requests
from langchain_core.embeddings import Embeddings

URL = "http://127.0.0.1:8000/embed"


class LocalAPIEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(URL, json={"inputs": texts}, timeout=120)
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([f"query: {text}"])[0]
