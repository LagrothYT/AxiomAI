import os
import json
import glob
import sys
import argparse
import torch

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tokenizer.bpe import BPETokenizer

def load_tokenizer():
    tokenizer_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Run train_tokenizer.py first.")
        sys.exit(1)
    return BPETokenizer.load(tokenizer_path)

def process_pretrain(tokenizer):
    print("Processing data for pretraining...")
    jsonl_files = glob.glob(os.path.join(config.data_dir, "*.jsonl"))
    
    from tqdm import tqdm
    total_size = sum(os.path.getsize(f) for f in jsonl_files)
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Prep: Pretrain")
    
    all_tokens = []
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                try:
                    data = json.loads(line)
                    text = ""
                    for turn in data.get("conversations", []):
                        text += turn.get("value", "") + " "
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                except json.JSONDecodeError:
                    continue
    pbar.close()
                    
    # Chunk into max_seq_len + 1 (for input and target)
    chunk_size = config.max_seq_len

    # Pad at the end to avoid dropping the last chunk
    rem = (len(all_tokens) - 1) % chunk_size
    if rem > 0:
        pad_len = chunk_size - rem
        all_tokens.extend([0] * pad_len)
        
    samples = []
    for i in range(0, len(all_tokens) - chunk_size + 1, chunk_size):
        samples.append(all_tokens[i:i + chunk_size + 1])
        
    if not samples:
        print("Error: Not enough data for pretraining.")
        return
        
    samples_tensor = torch.tensor(samples, dtype=torch.long)
    output_path = os.path.join(config.processed_data_dir, "pretrain.pt")
    torch.save(samples_tensor, output_path)
    print(f"Saved {len(samples)} pretraining samples to {output_path}")

def process_sft(tokenizer):
    print("Processing data for SFT...")
    jsonl_files = glob.glob(os.path.join(config.data_dir, "*.jsonl"))
    
    from tqdm import tqdm
    total_size = sum(os.path.getsize(f) for f in jsonl_files)
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Prep: SFT")
    
    all_samples = [] # List of (input_ids, loss_mask)
    
    # Roles for formatting
    human_prefix = f"{config.human_role}: "
    gpt_prefix = f"{config.gpt_role}: "
    eos_id = tokenizer.vocab.get(config.eos_token, 3)
    
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                try:
                    data = json.loads(line)
                    curr_ids = []
                    curr_mask = []
                    
                    for turn in data.get("conversations", []):
                        role = turn.get("from")
                        content = turn.get("value", "")
                        
                        if role == "human":
                            prefix_tokens = tokenizer.encode(human_prefix)
                            content_tokens = tokenizer.encode(content)
                            
                            turn_tokens = prefix_tokens + content_tokens
                            curr_ids.extend(turn_tokens)
                            curr_mask.extend([0] * len(turn_tokens)) # No loss on human turn
                        elif role == "gpt":
                            prefix_tokens = tokenizer.encode(gpt_prefix)
                            content_tokens = tokenizer.encode(content) + [eos_id]
                            
                            # No loss on "[GPT]: " prefix, only on content
                            curr_ids.extend(prefix_tokens)
                            curr_mask.extend([0] * len(prefix_tokens))
                            
                            curr_ids.extend(content_tokens)
                            curr_mask.extend([1] * len(content_tokens))
                            
                    # Truncate or pad to max_seq_len + 1 for input/target shifting
                    if len(curr_ids) > config.max_seq_len + 1:
                        curr_ids = curr_ids[:config.max_seq_len + 1]
                        curr_mask = curr_mask[:config.max_seq_len + 1]
                    elif len(curr_ids) < config.max_seq_len + 1:
                        pad_len = (config.max_seq_len + 1) - len(curr_ids)
                        curr_ids.extend([0] * pad_len) # Assuming <PAD> is 0
                        curr_mask.extend([0] * pad_len)
                        
                    all_samples.append((curr_ids, curr_mask))
                except json.JSONDecodeError:
                    continue
    pbar.close()
                    
    if not all_samples:
        print("Error: No samples found for SFT.")
        return
        
    ids_tensor = torch.tensor([s[0] for s in all_samples], dtype=torch.long)
    mask_tensor = torch.tensor([s[1] for s in all_samples], dtype=torch.float)
    
    output_path = os.path.join(config.processed_data_dir, "sft.pt")
    torch.save({"ids": ids_tensor, "mask": mask_tensor}, output_path)
    print(f"Saved {len(all_samples)} SFT samples to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], required=True)
    args = parser.parse_args()
    
    config.ensure_dirs()
    tokenizer = load_tokenizer()
    
    if args.mode == "pretrain":
        process_pretrain(tokenizer)
    else:
        process_sft(tokenizer)

if __name__ == "__main__":
    main()
