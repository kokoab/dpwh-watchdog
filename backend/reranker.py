import torch
from sentence_transformers import CrossEncoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=device
)


def rerank(query: str, candidates: list[dict], top_k: int = 50) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, c["chunk_text"]) for c in candidates]

    scores = reranker.predict(pairs, show_progress_bar=False)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        metadata_score = float(candidate.get("metadata_score", 0) or 0)
        candidate["final_rank_score"] = candidate["rerank_score"] + (0.12 * metadata_score)

    ranked = sorted(candidates, key=lambda x: x["final_rank_score"], reverse=True)

    return ranked[:top_k]
