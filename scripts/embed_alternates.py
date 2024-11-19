import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from src.embedder import Embedder
from utils.embed import batch_embed

def main():
    parser = argparse.ArgumentParser(description="Embed alternate descriptions using a specified model.")
    parser.add_argument('--model_path', type=str, help='Path to the model to use for embedding')
    args = parser.parse_args()

    with open('./data/emoji-info/alternate-descriptions.json', 'r') as file:
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

    embedder = Embedder(model_path=args.model_path)
    if not embedder.load_locally and embedder.is_hf_model:
        print("Loading API model")
        while True:
            test_input = ["test"]
            test_embeddings = embedder.embed(test_input)
            if test_embeddings.size > 0:
                break
        time.sleep(1)
        print("API model loaded")

    data_path = f'./data/embeddings/{embedder.model_path}'
    os.makedirs(data_path, exist_ok=True)
    alternate_embs_path = f'{data_path}/alternate_embeddings.npy'

    batch_size = 32
    embeddings = []
    for idx, inputs in enumerate(tqdm(inputs_lists, desc="Embedding inputs")):
        embeddings.append(batch_embed(inputs, embedder))
        # Save periodically and at end
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(inputs_lists):
            embeddings = np.stack(embeddings, axis=0)
            try:
                existing_embeddings = np.load(alternate_embs_path, allow_pickle=True)
                embeddings = np.concatenate((existing_embeddings, embeddings), axis=0)
            except:
                pass
            np.save(alternate_embs_path, embeddings)
            embeddings = []

if __name__ == '__main__':
    main()