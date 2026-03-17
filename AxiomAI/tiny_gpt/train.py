import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
    
    # Step-based scheduler: T_0 is steps per cycle (default 10 epochs worth)
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=config.lr * 0.1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    ppl_label = "SFT-PPL" if args.mode == "sft" else "PPL"
    
    for epoch in range(config.epochs):
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
                optimizer.zero_grad()
                
            train_loss += loss.item() * config.grad_accum_steps
            
            # Linear Warmup (first 100 steps)
            curr_step = epoch * steps_per_epoch + i
            if curr_step < 100:
                lr_scale = min(1.0, float(curr_step + 1) / 100.0)
                for group in optimizer.param_groups:
                    group['lr'] = lr_scale * config.lr
            else:
                scheduler.step(epoch + i / steps_per_epoch)
                
            loop.set_postfix(
                loss=f"{loss.item() * config.grad_accum_steps:.4f}",
                ppl=f"{ppl_label}: {torch.exp(torch.tensor(loss.item() * config.grad_accum_steps)):.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
            
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
        
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        
        print(f"Epoch {epoch+1}: Loss [T: {avg_train_loss:.4f}, V: {avg_val_loss:.4f}] | {ppl_label} [T: {train_ppl:.2f}, V: {val_ppl:.2f}]")
        
        # Sample Generation
        if config.train_samples > 0:
            model.eval()
            with torch.no_grad():
                prompt = config.train_sample_prompt
                if not prompt:
                    prompt = f"{config.human_role}: "
                
                input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                print(f"\n--- Samples (Epoch {epoch+1}) ---")
                for s in range(config.train_samples):
                    # Use defaults from config for temperature/top_p during training samples
                    output_ids, _, _ = model.generate(
                        input_ids, 
                        max_new_tokens=config.train_sample_len,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        eos_id=tokenizer.vocab.get(config.eos_token)
                    )
                    decoded = tokenizer.decode(output_ids[0].tolist())
                    print(f" Sample {s+1}: {decoded}")
                print("-" * 30 + "\n")
        
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
