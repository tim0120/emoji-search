from json import dumps
from urllib.parse import unquote

from src.embedder import Embedder

embedder = Embedder()

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
        results = embedder.k_nearest(query_embedding, 10, emb_type)
        
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