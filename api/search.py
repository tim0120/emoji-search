from json import dumps
from typing import List
from urllib.parse import unquote

import torch

from src.embed import Embedder
from utils.embed import k_nearest

def handler(request):
    query = request.query.get("query")
    emb_type = request.query.get("emb_type", "alternates")
    
    if not query:
        return {
            "statusCode": 400,
            "body": "Missing query parameter"
        }

    query = unquote(query)

    with open('./data/characters.txt', 'r', encoding='utf-8') as file:
        emoji_characters = file.read().splitlines()

    def get_k_nearest(queries: List[str], k: int, emb_type: str) -> List[List[str]]:
        embedder = Embedder()
        query_embeddings = embedder.embed(queries)

        if emb_type == 'unicodeName':
            embs_path = './data/unicodeName_embeddings.pt'
        elif emb_type == 'alternates':
            embs_path = './data/alternate_embeddings.pt'
        else:
            raise NotImplementedError(f"Embeddings type {emb_type} not supported")
        embeddings = torch.load(embs_path, weights_only=True)
        nearest_idxs = k_nearest(embeddings, query_embeddings, k)

        nearest_character_lists = []
        for query_idx in range(len(queries)):
            nearest_characters = [emoji_characters[idx].strip('"\'') for idx in nearest_idxs[:, query_idx]]
            nearest_character_lists.append(nearest_characters)
        return nearest_character_lists

    emojis = get_k_nearest([query], 10, emb_type)
    
    return {
        "statusCode": 200,
        "body": dumps(emojis),
        "headers": {
            "Content-Type": "application/json"
        }
    }