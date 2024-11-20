from typing import List
from numpy import ndarray, load, dot, einsum, argsort
from utils.inference import api_embed

try:
	from transformers import AutoModel, AutoTokenizer
	from utils.inference import local_embed
except:
	AutoModel = None
	AutoTokenizer = None
	local_embed = None


class Embedder:
	def __init__(
		self,
		model_path: str,
		load_locally: bool = False,
		is_hf_model: bool = True,
		load_emojis: bool = True
	):
		self.model_path = model_path
		self.load_locally = load_locally
		self.is_hf_model = is_hf_model
		if load_locally:
			self.tokenizer = AutoTokenizer.from_pretrained(model_path)
			self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
		self.embeddings = load(
			f'./data/embeddings/{model_path}/unicodeName_embeddings_quantized.npz',
			allow_pickle=True
		)['embeddings']
		if load_emojis:
			with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as file:
				self.emoji_characters = file.read().splitlines()

	def embed(self, input_texts: List[str]) -> ndarray:
		if self.load_locally:
			embeddings = local_embed(input_texts, self.tokenizer, self.model)
		else:
			embeddings = api_embed(input_texts, self.model_path, self.is_hf_model)
		return embeddings

	def k_nearest(self, queries: ndarray, k: int) -> List[int]:
		if len(self.embeddings.shape) == 2:
			similarities = dot(self.embeddings, queries.T)
		elif len(self.embeddings.shape) == 3:
			similarities = einsum('ijk,lk->il', self.embeddings, queries)
		else:
			raise NotImplementedError('Only 2D and 3D embeddings are supported')
		topk_indices = argsort(-similarities, axis=0)[:k]
		return topk_indices.flatten().tolist()