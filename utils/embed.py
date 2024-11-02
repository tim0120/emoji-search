from typing import List, Optional

import torch

from src.embed import Embedder

def batch_embed(inputs: List[str], embedder: Optional[Embedder] = None, bs: int = 16) -> torch.Tensor:
    if not embedder:
        embedder = Embedder()
    embeddings = []
    for i in range(0, len(inputs), bs):
        embeddings.append(embedder.embed(inputs[i:i+bs]))
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def k_nearest(embeddings: torch.Tensor, queries: torch.Tensor, k: int) -> torch.Tensor:
    if len(embeddings.shape) == 2:
        similarities = torch.einsum('ij,kj->ik', embeddings, queries)
        topk = torch.topk(similarities, k=k, dim=0)
        return topk.indices
    elif len(embeddings.shape) == 3:
        similarities = torch.einsum('ijk,lk->il', embeddings, queries)
        topk = torch.topk(similarities, k=k, dim=0)
        return topk.indices
    else:
        raise NotImplementedError("Only 2D and 3D embeddings are supported")