from typing import List

# Requires transformers>=4.36.0
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class Embedder:
    def __init__(
        self,
        model_path: str = 'Alibaba-NLP/gte-base-en-v1.5' # embedding dim: 768
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def embed(self, input_texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings