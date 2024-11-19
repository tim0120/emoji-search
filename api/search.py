from json import dumps
from urllib.parse import unquote

from src.embedder import Embedder

embedder = Embedder(model_path="mixedbread-ai/mxbai-embed-large-v1")
emoji_characters = open('./data/emoji-info/characters.txt', 'r', encoding='utf-8').read().splitlines()

def handler(request):
    query = request.query.get("query")
    emb_type = request.query.get("emb_type", "alternates")
    
    if not query:
        return {
            "statusCode": 400,
            "body": "Missing query parameter"
        }
    
    if emb_type not in ['unicodeName', 'alternates']:
        return {
            "statusCode": 400,
            "body": f"Unsupported embedding type: {emb_type}"
        }
    
    try:
        query = unquote(query)
        query_embedding = embedder.embed(query)
        idxs = embedder.k_nearest(query_embedding, 10, emb_type)
        results = [emoji_characters[idx] for idx in idxs]
        
        return {
            "statusCode": 200,
            "body": dumps(results),
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "s-maxage=3600"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": str(e)
        }