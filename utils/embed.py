from typing import List

import torch

from src.embed import Embedder

def batch_embed(inputs: List[str], bs=16) -> torch.Tensor:
    embedder = Embedder()
    embeddings = []
    for i in range(0, len(inputs), bs):
        embeddings.append(embedder.embed(inputs[i:i+bs]))
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def k_nearest(embeddings: torch.Tensor, queries: torch.Tensor, k: int) -> List[int]:
    similarities = torch.einsum('ij,kj->ik', embeddings, queries)
    topk = torch.topk(similarities, k=k, dim=0)
    return topk.indices