import json
import numpy as np
import configparser
import os
import random
from tokenizer.my_tokenizer import CharTokenizer

def preprocess_data():
    config = configparser.ConfigParser()
    config.read('configs/sft_config.ini')

    jsonl_path = config['DATA']['raw_data_path']
    vocab_path = config['DATA']['vocab_path']
    train_path = config['DATA']['train_path']
    val_path = config['DATA']['val_path']
    
    # New: companion loss-mask paths (auto-derived)
    train_mask_path = train_path.replace('.npy', '_mask.npy')
    val_mask_path = val_path.replace('.npy', '_mask.npy')

    tokenizer = CharTokenizer()
    if not tokenizer.load(vocab_path):
        print("Error: Tokenizer not found. Train it first from the main menu.")
        return

    print(f"Loading conversational data from {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    all_text_parts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                conv_turns = []
                for msg in data.get('conversations', []):
                    role = msg.get('from', 'unknown')
                    value = msg.get('value', '')
                    conv_turns.append((role, value))
                if conv_turns:
                    all_text_parts.append(conv_turns)
            except json.JSONDecodeError:
                pass

    if not all_text_parts:
        print("Error: No valid conversational data found.")
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

    # Encode each conversation + build loss mask
    # Mask = 1 ONLY on GPT assistant response tokens (including EOS)
    # Mask = 0 on everything else (human messages, role prefixes, BOS, etc.)
    def encode_parts(comps):
        all_tokens = []
        all_masks = []
        for conv in comps:
            all_tokens.append(tokenizer.bos_id)
            all_masks.append(0)  # BOS is ignored

            for role, value in conv:
                if role == 'gpt':
                    # Encode the FULL line as one unit so BPE merges match inference-time encoding
                    full_line = f"<{role}>: {value}\n"
                    full_tokens = tokenizer.encode(full_line)

                    # Measure how many tokens the prefix occupies when encoded in isolation
                    # to find the mask boundary (prefix = ignored, rest = trained)
                    prefix_len = len(tokenizer.encode(f"<{role}>: "))

                    all_tokens.extend(full_tokens)
                    all_masks.extend([0] * min(prefix_len, len(full_tokens)))
                    all_masks.extend([1] * max(0, len(full_tokens) - prefix_len))

                    all_tokens.append(tokenizer.eos_id)
                    all_masks.append(1)                          # train on EOS (model learns when to stop)
                else:
                    text = f"<{role}>: {value}\n"
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                    all_masks.extend([0] * len(tokens))          # ignore human/system turns
        return all_tokens, all_masks

    print("Encoding training tokens + loss masks...")
    train_tokens, train_masks = encode_parts(train_parts)
    train_np = np.array(train_tokens, dtype=np.int64)
    train_mask_np = np.array(train_masks, dtype=np.int8)

    print("Encoding validation tokens + loss masks...")
    val_tokens, val_masks = encode_parts(val_parts)
    val_np = np.array(val_tokens, dtype=np.int64)
    val_mask_np = np.array(val_masks, dtype=np.int8)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    np.save(train_path, train_np)
    np.save(train_mask_path, train_mask_np)
    np.save(val_path, val_np)
    np.save(val_mask_path, val_mask_np)

    print(f"Conversations -> Train: {len(train_parts):,}  |  Val: {len(val_parts):,}")
    print(f"Saved {len(train_np):,} train tokens + mask and {len(val_np):,} val tokens + mask.")
    print("Preprocessing complete! (Loss masking is now active for SFT)")