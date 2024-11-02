import json
import os

import torch
from tqdm import tqdm

from src.embed import Embedder
from utils.embed import batch_embed

def main():
    with open('./data/alternate-descriptions.json', 'r') as file:
        alternates_json = json.load(file)
    inputs_lists = [
        # remove extraneous prefix and combine with alternates into a single list
        [" ".join(alt_dict['input'].split(' ')[1:])] + alt_dict['alternates']
        for alt_dict in alternates_json
    ]
    assert all(
        all(isinstance(alt, str) for alt in alt_list)
        for alt_list in inputs_lists
    )

    alternate_embs_path = './data/alternate_embeddings.pt'
    embedder = Embedder()
    batch_size = 32
    embeddings = []
    for idx, inputs in enumerate(tqdm(inputs_lists, desc="Embedding inputs")):
        embeddings.append(batch_embed(inputs, embedder))
        # Save periodically and at end
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(inputs_lists):
            embeddings = torch.stack(embeddings, dim=0)
            try:
                existing_embeddings = torch.load(alternate_embs_path, weights_only=True)
                embeddings = torch.cat((existing_embeddings, embeddings), dim=0)
            except:
                pass
            torch.save(embeddings, alternate_embs_path)
            embeddings = []

if __name__ == '__main__':
    main()