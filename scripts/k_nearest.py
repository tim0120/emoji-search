import argparse
from typing import List

import torch

from src.embed import Embedder
from utils.embed import k_nearest

def main():
    parser = argparse.ArgumentParser(description="Process one or more string inputs.")
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to find')
    args = parser.parse_args()

    with open('./data/emoji-characters.txt', 'r', encoding='utf-8') as file:
        emoji_characters = file.read().splitlines()

    def get_k_nearest(queries: List[str], k: int) -> List[List[str]]:
        embedder = Embedder()
        query_embeddings = embedder.embed(queries)

        embeddings = torch.load('./data/unicode_name_embeddings.pt', weights_only=True)
        nearest_idxs = k_nearest(embeddings, query_embeddings, k)

        nearest_character_lists = []
        for query_idx in range(len(queries)):
            nearest_characters = [emoji_characters[idx].strip('"\'') for idx in nearest_idxs[:, query_idx]]
            nearest_character_lists.append(nearest_characters)
        return nearest_character_lists

    while True:
        user_input = input("Enter queries separated by commas (press enter to quit): ")
        if user_input.strip() == '':
            break
        queries = [query.strip() for query in user_input.split(',')]
        nearest_character_lists = get_k_nearest(queries, args.k)

        for query_idx, nearest_characters in enumerate(nearest_character_lists):
            print(f"{queries[query_idx]}:", " ".join(nearest_characters))

if __name__ == '__main__':
    main()