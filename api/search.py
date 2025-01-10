from http.server import BaseHTTPRequestHandler
from json import dumps
from os import getenv
from time import sleep
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from numpy import argsort, array, dot, einsum, load
from requests import post
from requests.exceptions import RequestException

load_dotenv()

model_id = "openai/text-embedding-3-small"
model_name = model_id.split('/')[-1]
embeddings = load(
    f'./data/embeddings/{model_id}/unicodeNames.npz', allow_pickle=True
)['embeddings']
emoji_characters = open('./data/emoji-info/characters.txt', 'r', encoding='utf-8').read().splitlines()

def get_embeddings(text, max_retries=3, initial_delay=1):
    API_URL = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = post(
                API_URL,
                headers=headers,
                json={
                    "input": text,
                    "model": model_name
                }
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                return {"error": str(e)}
            sleep(delay)
            delay *= 2  # Exponential backoff

def k_nearest(queries, k):
    if len(embeddings.shape) == 2:
        similarities = dot(embeddings, queries.T)
    elif len(embeddings.shape) == 3:
        similarities = einsum('ijk,lk->il', embeddings, queries)
    else:
        raise NotImplementedError('Only 2D and 3D embeddings are supported')
    topk_indices = argsort(-similarities, axis=0)[:k]
    return topk_indices.flatten().tolist()

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
        response = get_embeddings(query)
        if isinstance(response, dict) and "error" in response:
            return {
                "statusCode": 500,
                "body": response["error"]
            }
        query_embedding = array(response)
        idxs = k_nearest(query_embedding, int(getenv('K_NEAREST')))
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