from typing import List, Optional, Union

import numpy as np

from src.embedder import Embedder

def batch_embed(inputs: List[str], embedder: Optional[Embedder] = None, bs: int = 16) -> np.ndarray:
	if not embedder:
		embedder = Embedder()
	embeddings = []
	for i in range(0, len(inputs), bs):
		embeddings.append(embedder.embed(inputs[i:i+bs]))
	embeddings = np.concatenate(embeddings, axis=0)
	return embeddings

def k_nearest(embeddings: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
	if len(embeddings.shape) == 2:
		similarities = np.dot(embeddings, queries.T)
		topk_indices = np.argsort(-similarities, axis=0)[:k]
		return topk_indices
	elif len(embeddings.shape) == 3:
		similarities = np.einsum('ijk,lk->il', embeddings, queries)
		topk_indices = np.argsort(-similarities, axis=0)[:k]
		return topk_indices
	else:
		raise NotImplementedError("Only 2D and 3D embeddings are supported")