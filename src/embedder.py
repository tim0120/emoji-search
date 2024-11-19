from typing import List

import numpy as np
from transformers import AutoModel, AutoTokenizer

from utils.inference import local_embed, api_embed

class Embedder:
	def __init__(
		self,
		model_path: str,
		load_locally: bool = False,
		is_hf_model: bool = True
	):
		self.model_path = model_path
		self.load_locally = load_locally
		self.is_hf_model = is_hf_model
		if load_locally:
			self.tokenizer = AutoTokenizer.from_pretrained(model_path)
			self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

	def embed(self, input_texts: List[str]) -> np.ndarray:
		if self.load_locally:
			embeddings = local_embed(input_texts, self.tokenizer, self.model)
		else:
			embeddings = api_embed(input_texts, self.model_path, self.is_hf_model)
		return embeddings