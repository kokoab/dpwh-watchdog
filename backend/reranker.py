import torch
from sentence_transformers import CrossEncoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=device
)


def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, c["chunk_text"]) for c in candidates]

    scores = rerank.predict(pairs, show_progress_bar=False)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return ranked[:top_k]
