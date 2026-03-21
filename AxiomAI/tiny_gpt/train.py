import os
import sys
import argparse
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import copy

import config
from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], required=True)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config.ensure_dirs()
    
    # Global deterministic seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
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
        # data is {"ids": [N, max_seq_len + 1], "mask": [N, max_seq_len + 1]}
        ids = data["ids"]
        # targets are shifted ids
        x = ids[:, :-1]
        y = ids[:, 1:]
        # mask needs to match shifted targets
        mask = data["mask"][:, 1:]
        dataset = TensorDataset(x, y, mask)
        
    # Split data
    if len(dataset) < 2:
        print("Warning: Dataset too small for validation split. Using all data for training.")
        train_ds = dataset
        val_ds = dataset
    else:
        train_size = int((1.0 - config.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        # Ensure at least one validation sample if we have data to spare
        if val_size == 0:
            train_size -= 1
            val_size = 1
        torch.manual_seed(config.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    # Separate parameters into decay and no-decay groups (exclude biases and LN)
    param_dict = {n: p for n, p in model.named_parameters()}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer = optim.AdamW(param_groups, lr=config.lr)
    
    # Global Step-based LR Scheduler
    total_steps = config.epochs * math.ceil(len(train_loader) / config.grad_accum_steps)
    warmup_steps = config.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        # Cosine decay from 1.0 down to min_lr ratio
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return config.min_lr + (1.0 - config.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    start_epoch = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_val_loss = checkpoint.get("best_val_loss", float('inf'))
                print(f"Resumed at epoch {start_epoch} with val_loss {best_val_loss:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded bare weights. Starting from epoch 0.")
        else:
            print(f"Error: Resume checkpoint {args.resume} not found.")
            sys.exit(1)
            
    ppl_label = "SFT-PPL" if args.mode == "sft" else "PPL"
    
    save_path = config.best_model_path.replace("best.pt", f"{args.mode}_best.pt")
    
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", leave=False)
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
                
                if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    scheduler.step() # Moved scheduler.step() here
                    optimizer.zero_grad()
                    
                train_loss += loss.item() * config.grad_accum_steps
                
                loop.set_postfix(
                    loss=f"{loss.item() * config.grad_accum_steps:.4f}",
                    ppl=f"{torch.exp(torch.tensor(loss.item() * config.grad_accum_steps)):.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if args.mode == "pretrain":
                        x_b, y_b = [t.to(device) for t in batch]
                        _, loss = model(x_b, y_b)
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
                    total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / max(1, len(val_loader))
            
            train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
            val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
            
            is_best = avg_val_loss < best_val_loss
            best_marker = " [BEST]" if is_best else ""
            
            # Log metrics
            print(f"Epoch {epoch+1}: Loss [T: {avg_train_loss:.4f}, V: {avg_val_loss:.4f}] | {ppl_label} [T: {train_ppl:.2f}, V: {val_ppl:.2f}] | LR: {optimizer.param_groups[0]['lr']:.6f}{best_marker}")
            
            # Check for improvement and save best model during training
            stop_training = False # Initialize for this epoch
            if is_best:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({
                    "model": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss
                }, save_path) 
                print(f"  --> New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Early stopping check
            if config.early_stopping_enabled and epochs_no_improve >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                stop_training = True
                
            if stop_training:
                break
                    
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
        interrupted_path = save_path.replace(".pt", "_interrupted.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch if 'epoch' in locals() else start_epoch,
            "best_val_loss": best_val_loss
        }, interrupted_path)
        print(f"Current model state saved to {interrupted_path}")

    # Final cleanup after training loop
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nTraining complete. Best model loaded with Val Loss: {best_val_loss:.4f}")
    else:
        # Fallback if no validation happened or no improvement (e.g., very short training)
        # In this case, the last model state is saved if no better one was found.
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch if 'epoch' in locals() else 0,
            "best_val_loss": best_val_loss
        }, save_path)
        print(f"\nTraining complete. No improvement found, last model state saved to {save_path}.")


if __name__ == "__main__":
    train()
