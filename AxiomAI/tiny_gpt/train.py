import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

import config
from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], required=True)
    args = parser.parse_args()
    
    config.ensure_dirs()
    device = torch.device("cpu")
    if os.environ.get("USE_ROCM") == "1":
        # Support ROCm if requested and available via torch.cuda (ROCm often maps to cuda)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using ROCm (CUDA-mapped) device")
    
    print(f"Using device: {device}")
    
    tokenizer_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Run train_tokenizer.py first.")
        sys.exit(1)
    # Tokenizer is loaded to ensure we match the trained vocab size
    tokenizer = BPETokenizer.load(tokenizer_path)
    current_vocab_size = len(tokenizer.vocab)
    
    model = TinyGPT(
        vocab_size=current_vocab_size, # Use actual size from trained tokenizer
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(device)
    
    # Load processed data
    data_path = os.path.join(config.processed_data_dir, f"{args.mode}.pt")
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}. Run prepare_data.py first.")
        sys.exit(1)
        
    if args.mode == "pretrain":
        data = torch.load(data_path, weights_only=True)
        # data is [N, max_seq_len + 1]
        x = data[:, :-1]
        y = data[:, 1:]
        dataset = TensorDataset(x, y)
    else:
        data = torch.load(data_path, weights_only=True)
        # data is {"ids": [N, max_seq_len], "mask": [N, max_seq_len]}
        ids = data["ids"]
        # targets are shifted ids
        x = ids[:, :-1]
        y = ids[:, 1:]
        # mask needs to match shifted targets
        mask = data["mask"][:, 1:]
        dataset = TensorDataset(x, y, mask)
        
    # Split data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Guard against 0 validation size
    if val_size == 0 and len(dataset) > 1:
        train_size = len(dataset) - 1
        val_size = 1
    elif val_size == 0:
        print("Warning: Dataset too small for validation split. Using all data for training.")
        train_ds = dataset
        val_ds = dataset # Just to avoid crash, though it will overfit
    else:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for i, batch in enumerate(loop):
            if args.mode == "pretrain":
                x_b, y_b = [t.to(device) for t in batch]
                logits, loss = model(x_b, y_b)
            else:
                x_b, y_b, m_b = [t.to(device) for t in batch]
                logits, _ = model(x_b)
                # Custom loss for SFT with mask
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y_b.view(-1), 
                    reduction='none'
                )
                loss = (loss * m_b.view(-1)).sum() / (m_b.sum() + 1e-9)
            
            loss = loss / config.grad_accum_steps
            loss.backward()
            
            if (i + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item() * config.grad_accum_steps
            loop.set_postfix(loss=loss.item() * config.grad_accum_steps)
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.mode == "pretrain":
                    x_b, y_b = [t.to(device) for t in batch]
                    _, loss = model(x_b, y_b)
                else:
                    x_b, y_b, m_b = [t.to(device) for t in batch]
                    logits, _ = model(x_b)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        y_b.view(-1), 
                        reduction='none'
                    )
                    loss = (loss * m_b.view(-1)).sum() / (m_b.sum() + 1e-9)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.best_model_path)
            print(f"Saved best model to {config.best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
