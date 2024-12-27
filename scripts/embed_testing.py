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

embedder = Embedder(model_id='text-embedding-3-small')
embeddings = embedder.embed(input_texts)
breakpoint()