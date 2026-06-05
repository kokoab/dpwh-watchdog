import torch
from sentence_transformers import CrossEncoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=device
)

    
