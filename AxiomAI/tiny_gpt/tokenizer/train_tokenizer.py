import os
import json
import glob
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tokenizer.bpe import BPETokenizer

def load_sharegpt_texts(data_dir):
    texts = []
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {data_dir}")
        print("Please drop your ShareGPT format data into the data/ directory.")
        sys.exit(1)
        
    for file_path in jsonl_files:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    for turn in data.get("conversations", []):
                        texts.append(turn.get("value", ""))
                except json.JSONDecodeError:
                    continue
    return texts

def main():
    config.ensure_dirs()
    
    print("Collecting texts for tokenizer training...")
    texts = load_sharegpt_texts(config.data_dir)
    
    if not texts:
        print("Error: No conversation text found in data files.")
        sys.exit(1)
        
    print(f"Collected {len(texts)} turns. Training BPE tokenizer (target vocab size: {config.vocab_size})...")
    
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(texts)
    
    tokenizer_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    main()
