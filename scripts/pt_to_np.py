import argparse
import os
from scripts.train_head import EmojiHead
import numpy as np
import torch

def convert_to_numpy(model):
    numpy_weights = {}
    layer_idx = 0
    
    for i in range(len(model.mlp)):
        layer = model.mlp[i]
        
        # Handle Linear layers
        if isinstance(layer, torch.nn.Linear):
            layer_name = f'layer{layer_idx + 1}'
            numpy_weights[layer_name] = {
                'weight': layer.weight.data.detach().numpy(),
                'bias': layer.bias.data.detach().numpy()
            }
            
            # Check if next layer is BatchNorm
            if i + 1 < len(model.mlp) and isinstance(model.mlp[i + 1], torch.nn.BatchNorm1d):
                bn_layer = model.mlp[i + 1]
                numpy_weights[layer_name].update({
                    'bn_mean': bn_layer.running_mean.detach().numpy(),
                    'bn_var': bn_layer.running_var.detach().numpy(),
                    'bn_weight': bn_layer.weight.detach().numpy(),
                    'bn_bias': bn_layer.bias.detach().numpy()
                })
            
            layer_idx += 1
            
    return numpy_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='wandb/latest-run/files/best_emoji_head.pt', help='Path to .pt model file')
    parser.add_argument('--output_dir', type=str, default='data/deployment/', help='Directory to save numpy weights')
    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.model_path}")
    # text-embedding-3-small has dim 1536
    # there are 1859 emojis in data/characters.txt
    model = EmojiHead(input_dim=1536, num_classes=1859)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Convert to numpy
    print("Converting weights to numpy")
    numpy_weights = convert_to_numpy(model)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save numpy weights
    output_path = os.path.join(args.output_dir, 'model_weights.npz')
    print(f"Saving weights to {output_path}")
    np.savez(output_path, **numpy_weights)

if __name__ == '__main__':
    main()