import json
import numpy as np
import configparser
import os
import random
from tokenizer.my_tokenizer import CharTokenizer

def preprocess_data():
    config = configparser.ConfigParser()
    config.read('configs/config.ini')

    raw_path = config['DATA']['raw_data_path']
    vocab_path = config['DATA']['vocab_path']
    train_path = config['DATA']['train_path']
    val_path = config['DATA']['val_path']

    tokenizer = CharTokenizer()
    if not tokenizer.load(vocab_path):
        print("Error: Tokenizer not found. Train it first from the main menu.")
        return

    print(f"Scanning data from {raw_path}...")
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    files_to_process = []
    if os.path.isdir(raw_path):
        for f in os.listdir(raw_path):
            files_to_process.append(os.path.join(raw_path, f))
    else:
        files_to_process.append(raw_path)

    all_text_parts = []
    
    for f_path in files_to_process:
        if f_path.endswith('.jsonl'):
            print(f"  [JSONL] Parsing ShareGPT structures: {os.path.basename(f_path)}")
            with open(f_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        conv_turns = []
                        for msg in data.get('conversations', []):
                            role = msg.get('from', 'unknown')
                            value = msg.get('value', '')
                            conv_turns.append((role, value))
                        if conv_turns:
                            all_text_parts.append(('conv', conv_turns))
                    except json.JSONDecodeError:
                        pass
        elif f_path.endswith('.txt'):
            print(f"  [TXT]   Parsing raw text: {os.path.basename(f_path)}")
            with open(f_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    all_text_parts.append(('raw', text))
        else:
            print(f"  [Skip]  Unsupported format: {os.path.basename(f_path)}")

    if not all_text_parts:
        print("Error: No valid data found in the target path.")
        return

    # Shuffle conversations (not tokens!) to reduce ordering bias
    random.shuffle(all_text_parts)

    # --- Split conversations BEFORE encoding to prevent data leakage ---
    split_idx = int(len(all_text_parts) * 0.9)
    split_idx = max(1, split_idx)

    train_parts = all_text_parts[:split_idx]
    val_parts = all_text_parts[split_idx:]

    if not val_parts:
        val_parts = [all_text_parts[-1]]
        print("Warning: Dataset too small for a clean split. Duplicated last conversation into val set.")

    # Encode each conversation wrapped in <BOS>, and properly cap <gpt>
    # responses with <EOS> so the model explicitly learns to stop talking.
    def encode_parts(comps):
        all_tokens = []
        for ptype, content in comps:
            all_tokens.append(tokenizer.bos_id)
            if ptype == 'conv':
                for role, value in content:
                    text = f"<{role}>: {value}\n"
                    all_tokens.extend(tokenizer.encode(text))
                    if role == 'gpt':
                        all_tokens.append(tokenizer.eos_id)
            else:
                # Raw text encoding
                all_tokens.extend(tokenizer.encode(content))
                all_tokens.append(tokenizer.eos_id)
        return all_tokens

    print("Encoding training tokens...")
    train_tokens = encode_parts(train_parts)
    train_np = np.array(train_tokens, dtype=np.int64)

    print("Encoding validation tokens...")
    val_tokens = encode_parts(val_parts)
    val_np = np.array(val_tokens, dtype=np.int64)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    np.save(train_path, train_np)
    np.save(val_path, val_np)

    print(f"Conversations -> Train: {len(train_parts):,}  |  Val: {len(val_parts):,}")
    print(f"Saved {len(train_np):,} train tokens and {len(val_np):,} val tokens.")
    print("Preprocessing complete!")