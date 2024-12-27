import argparse
import os

from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

from src.embedder import Embedder

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Embed strings from a text file')
    parser.add_argument('--input_file', type=str, default='data/emoji-info/unicodeNames.txt', help='Path to input text file')
    parser.add_argument('--model_id', type=str, default='openai/text-embedding-3-small', help='Model ID to use for embedding')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file '{args.input_file}' does not exist.")

    embedder = Embedder(model_id=args.model_id)
    with open(args.input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    embeddings = embedder.embed(lines, use_tqdm=True)
            
    text_type = args.input_file.split('/')[-1].split('.')[0]
    save_dir = f'data/embeddings/{args.model_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/{text_type}.npz'
    np.savez(save_path, embeddings=embeddings)
    if os.path.exists(save_path):
        print(f"Embeddings saved to '{save_path}'")
    else:
        print(f"Failed to save embeddings to '{save_path}'")

if __name__ == '__main__':
    main()