from typing import List, Optional

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