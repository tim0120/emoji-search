import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EmojiDataset(Dataset):
    def __init__(self, jsonl_path, emoji_to_index):
        self.jsonl_path = jsonl_path
        self.emoji_to_index = emoji_to_index
        self.num_classes = len(emoji_to_index)
        # Calculate and store length during initialization
        self.length = sum(1 for line in open(self.jsonl_path, 'r', encoding='utf-8')) - 1  # -1 for metadata line
        print(f"Initialized EmojiDataset with {self.num_classes} classes and {self.length} samples")
        
        # Load all data into memory
        self.data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip metadata
            for line in f:
                data = json.loads(line)
                if 'embedding' in data and 'emoji_indices' in data:
                    self.data.append(data)
    
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        data = self.data[idx]
        embedding = torch.tensor(data['embedding'], dtype=torch.float32)
        
        # Create one-hot labels for emoji and similar emojis
        labels = torch.zeros(self.num_classes)
        for idx in data['emoji_indices']:
            labels[idx] = 1.0
        
        # Apply label smoothing
        alpha = 0.1  # Label smoothing factor
        smoothed_labels = (1 - alpha) * labels + alpha / self.num_classes
        
        return embedding, smoothed_labels

class EmojiHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        print(f"Initializing EmojiHead with input_dim={input_dim}, num_classes={num_classes}")
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

def main():
    print("Starting emoji head training script")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    # Initialize wandb
    print("Initializing wandb")
    wandb.init(project="emoji-head", config=args)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load emoji mapping
    print("Loading emoji mapping")
    with open('./data/emoji-info/characters.txt', 'r', encoding='utf-8') as f:
        emoji_characters = f.read().splitlines()
    emoji_to_index = {emoji: idx for idx, emoji in enumerate(emoji_characters)}
    print(f"Loaded {len(emoji_characters)} emoji characters")

    # Initialize model
    print("Initializing model, optimizer and criterion")
    model = EmojiHead(input_dim=1536, num_classes=len(emoji_to_index)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Add scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    print("Initialized learning rate scheduler")
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'best_emoji_head.pt'
    print(f"Model will be saved to: {best_model_path}")

    # Create datasets
    print("Creating datasets")
    full_dataset = EmojiDataset('./data/synthetic_generation_output.jsonl', emoji_to_index)
    
    train_size = int(len(full_dataset) * args.train_split)
    test_size = len(full_dataset) - train_size
    
    print("Splitting dataset into train and test sets")
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")

    print("Starting training loop")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # Training
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        print("Training phase:")
        train_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} training")
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            
            # Log batch loss
            wandb.log({"batch_loss": loss.item()})
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_pbar.update()
        train_pbar.close()
        
        print("Validation phase:")
        # Validation
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        val_pbar = tqdm(total=len(test_loader), desc=f"Epoch {epoch+1} validation")
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                # Add prediction metrics
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == labels).float().sum()
                total_predictions += labels.numel()
                
                running_val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                val_pbar.update()
        val_pbar.close()
        
        train_loss = running_train_loss / train_batches
        val_loss = running_val_loss / val_batches
        accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch + 1} Results:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Accuracy: {accuracy:.4f}')
        
        # Log metrics
        print("Logging metrics to wandb")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Learning rate scheduling
        print("Updating learning rate scheduler")
        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {val_loss:.4f}, saving model")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    print("Training completed")
    print("Loading best model and finishing up")
    model.load_state_dict(torch.load(best_model_path))
    wandb.save(best_model_path)
    wandb.finish()

if __name__ == '__main__':
    main()
