import argparse
import os
from typing import List

import numpy as np

from src.embedder import Embedder

def main():
    parser = argparse.ArgumentParser(description="Process one or more string inputs.")
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to find')
    parser.add_argument('--emb_type', type=str, default='unicodeNames', help='Type of embeddings to use')
    parser.add_argument('--model_id', type=str, default='openai/text-embedding-3-small', help='Path to the model to use for embedding')
    parser.add_argument('--quantized', type=bool, default=True, help='Use quantized embeddings if they exist')
    parser.add_argument('--use_model', action='store_true', help='Use trained model to predict emoji probabilities')
    parser.add_argument('--model_path', type=str, default='./wandb/latest-run/files/best_emoji_head.pt', help='Path to trained model weights')
    args = parser.parse_args()

    with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as file:
        emoji_characters = file.read().splitlines()

    embedder = Embedder(args.model_id)

    model = None
    if args.use_model:
        import torch
        import torch.nn as nn
        
        class EmojiHead(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, num_classes),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.mlp(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EmojiHead(input_dim=1536, num_classes=len(emoji_characters)).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    def get_k_nearest_emojis(queries: List[str], k: int) -> List[List[str]]:
        query_embeddings = embedder.embed(queries)

        embs_path = f"./data/embeddings/{args.model_id}/{args.emb_type}.npz"
        if args.quantized and os.path.exists(embs_path.replace('.npz', '_quantized.npz')):
            embs_path = embs_path.replace('.npz', '_quantized.npz')
        embeddings = np.load(embs_path)['embeddings']
        
        if args.use_model:
            # Get model probabilities
            with torch.no_grad():
                query_tensor = torch.tensor(query_embeddings, dtype=torch.float32).to(device)
                probs = model(query_tensor).cpu().numpy()
            
            # Get embedding similarities 
            similarities = embedder.k_nearest(embeddings, query_embeddings, k, return_scores=True)
            
            # Combine scores for each query
            nearest_idxs = []
            for query_idx in range(len(queries)):
                query_probs = probs[query_idx]
                query_sims = similarities[:, query_idx]
                
                # Normalize probabilities and similarities to [0,1] range
                norm_probs = (query_probs - query_probs.min()) / (query_probs.max() - query_probs.min())
                norm_sims = (query_sims - query_sims.min()) / (query_sims.max() - query_sims.min())
                
                # Weighted average of normalized scores
                alpha = 0.9  # Weight for model probabilities vs embedding similarities
                combined_scores = alpha * norm_probs + (1 - alpha) * norm_sims
                query_nearest_idxs = np.argsort(-combined_scores)[:k]
                nearest_idxs.append(query_nearest_idxs)
            nearest_idxs = np.array(nearest_idxs)
        else:
            nearest_idxs = embedder.k_nearest(embeddings, query_embeddings, k)
        
        print('nearest:', nearest_idxs)

        nearest_character_lists = []
        for query_idx in range(len(queries)):
            query_nearest_idxs = nearest_idxs[query_idx]
            nearest_characters = [emoji_characters[idx].strip('"\'') for idx in query_nearest_idxs]
            nearest_character_lists.append(nearest_characters)
        return nearest_character_lists

    while True:
        user_input = input("Enter queries separated by commas (press enter to quit): ")
        if user_input.strip() == '':
            break
        queries = [query.strip() for query in user_input.split(',')]
        nearest_character_lists = get_k_nearest_emojis(queries, args.k)

        for query_idx, nearest_characters in enumerate(nearest_character_lists):
            print(f"{queries[query_idx]}:", " ".join(nearest_characters))
if __name__ == '__main__':
    main()