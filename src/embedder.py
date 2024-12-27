from typing import List

import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils.inference import api_embed, local_embed


class Embedder:
	def __init__(
		self,
		model_id: str,
		load_locally: bool = False,
		is_hf_model: bool = False,
		load_emojis: bool = True
	):
		self.model_id = model_id
		self.load_locally = load_locally
		self.is_hf_model = is_hf_model
		if load_locally:
			self.tokenizer = AutoTokenizer.from_pretrained(model_id)
			self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
		if load_emojis:
			with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as file:
				self.emoji_characters = file.read().splitlines()

	def embed(self, input_texts: List[str], batch_size: int = 16, use_tqdm: bool = False) -> np.ndarray:
		embeddings = []
		iterator = tqdm(range(0, len(input_texts), batch_size), desc="Embedding texts") if use_tqdm \
			else range(0, len(input_texts), batch_size)
		for i in iterator:
			batch = input_texts[i:i+batch_size]
			batch_embs = local_embed(batch, self.tokenizer, self.model) if self.load_locally \
				else api_embed(batch, self.model_id, self.is_hf_model)
			embeddings.append(batch_embs)
		return np.concatenate(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]

	def k_nearest(self, embeddings: np.ndarray, queries: np.ndarray, k: int) -> List[int]:
		if len(embeddings.shape) == 2:
			similarities = np.dot(embeddings, queries.T)
		elif len(embeddings.shape) == 3:
			similarities = np.einsum('ijk,lk->il', embeddings, queries)
		else:
			raise NotImplementedError('Only 2D and 3D embeddings are supported')
		topk_indices = np.argsort(-similarities, axis=0)[:k].T
		return topk_indices.tolist()