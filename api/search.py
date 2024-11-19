from http.server import BaseHTTPRequestHandler
from json import dumps
from urllib.parse import parse_qs, urlparse
from src.embedder import Embedder

embedder = Embedder(model_path="mixedbread-ai/mxbai-embed-large-v1")
emoji_characters = open('./data/emoji-info/characters.txt', 'r', encoding='utf-8').read().splitlines()

def handle_request(query_params):
    query = query_params.get('query', [None])[0]
    emb_type = query_params.get('emb_type', ['alternates'])[0]
    
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

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse query parameters
        query_components = parse_qs(urlparse(self.path).query)
        
        # Get response from handler
        response = handle_request(query_components)
        
        # Set response status code
        self.send_response(response['statusCode'])
        
        # Set headers
        if 'headers' in response:
            for header, value in response['headers'].items():
                self.send_header(header, value)
        self.end_headers()
        
        # Send response body
        self.wfile.write(response['body'].encode())