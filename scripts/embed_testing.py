from transformers import AutoModel, AutoTokenizer

from src.embedder import Embedder

input_texts = [
    "hey",
    "hi",
    "bye",
    "no",
    "yes",
    "dog",
    "cat"
]

embedder = Embedder()
embeddings = embedder.embed(input_texts)
breakpoint()