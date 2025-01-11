from http.server import BaseHTTPRequestHandler
from json import dumps
from os import getenv
from time import sleep
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from numpy import argsort, array, dot, load, maximum, sqrt, exp
from requests import post
from requests.exceptions import RequestException

load_dotenv()

model_name = 'text-embedding-3-small'
embeddings = load('./data/deployment/openai-unicodeNames-embeddings.npz', allow_pickle=True)['embeddings']
model_weights = load('./data/deployment/model_weights.npz', allow_pickle=True)
emoji_characters = open('./data/deployment/characters.txt', 'r', encoding='utf-8').read().splitlines()

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

def numpy_inference(x, weights):  # x shape: (input_dim,)
    # Layer 1
    x = dot(weights['layer1']['weight'], x) + weights['layer1']['bias']
    x = (x - weights['layer1']['bn_mean']) / sqrt(weights['layer1']['bn_var'] + 1e-5)
    x = weights['layer1']['bn_weight'] * x + weights['layer1']['bn_bias']
    x = maximum(0, x)  # ReLU
    
    # Layer 2
    x = dot(weights['layer2']['weight'], x) + weights['layer2']['bias']
    x = (x - weights['layer2']['bn_mean']) / sqrt(weights['layer2']['bn_var'] + 1e-5)
    x = weights['layer2']['bn_weight'] * x + weights['layer2']['bn_bias']
    x = maximum(0, x)  # ReLU
    
    # Layer 3
    x = dot(weights['layer3']['weight'], x) + weights['layer3']['bias']
    x = 1/(1 + exp(-x))  # Sigmoid
    
    return x

def get_nearest_idxs(query_embedding, k):
    probs = numpy_inference(query_embedding, model_weights)
    similarities = dot(embeddings, query_embedding.T)
    
    # Normalize probabilities and similarities to [0,1] range
    norm_probs = (probs - probs.min()) / (probs.max() - probs.min())
    norm_sims = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    
    # Weighted average of normalized scores
    alpha = 0.9  # Weight for model probabilities vs embedding similarities
    scores = alpha * norm_probs + (1 - alpha) * norm_sims
    
    topk_indices = argsort(-scores, axis=0)[:k]
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
        idxs = get_nearest_idxs(query_embedding, int(getenv('K_NEAREST')))
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