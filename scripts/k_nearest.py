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
    args = parser.parse_args()

    with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as file:
        emoji_characters = file.read().splitlines()

    embedder = Embedder(args.model_id)

    def get_k_nearest_emojis(queries: List[str], k: int) -> List[List[str]]:
        query_embeddings = embedder.embed(queries)

        embs_path = f"./data/embeddings/{args.model_id}/{args.emb_type}.npz"
        if args.quantized and os.path.exists(embs_path.replace('.npz', '_quantized.npz')):
                embs_path = embs_path.replace('.npz', '_quantized.npz')
        embeddings = np.load(embs_path)['embeddings']
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