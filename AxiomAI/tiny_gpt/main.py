import sys
import os
import subprocess
import json
import config

def format_size(size_bytes):
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB")
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_data_specs():
    data_path = os.path.join(config.data_dir, "data.jsonl")
    if not os.path.exists(data_path):
        return "[!] data.jsonl not found"

    raw_size = os.path.getsize(data_path)
    content_chars = 0
    conv_count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                conv_count += 1
                for conv in item.get("conversations", []):
                    content_chars += len(conv.get("value", ""))
            except: continue

    specs = [
        f"File:         {os.path.basename(data_path)} ({format_size(raw_size)})",
        f"Convos:       {conv_count:,}",
        f"Content:      {content_chars:,} chars"
    ]

    tok_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        try:
            from tokenizer.bpe import BPETokenizer
            tokenizer = BPETokenizer.load(tok_path)
            total_tokens = 0
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = ""
                        for conv in item.get("conversations", []):
                            text += f"{conv['from']}: {conv['value']} "
                        total_tokens += len(tokenizer.encode(text))
                    except: continue
            specs.append(f"Tokens:       {total_tokens:,}")
        except: pass

    return "\n  ".join(specs)

def run_command(command):
    try:
        # We use sys.executable to ensure we use the same python interpreter
        full_command = [sys.executable] + command
        print(f"Running: {' '.join(full_command)}")
        subprocess.run(full_command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error running command: {e}")
        return False



def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_status():
    status = []
    
    tok_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    tok_exists = os.path.exists(tok_path)
    status.append(f"  {'Tokenizer:':<18} [{'X' if tok_exists else ' '}]")
    
    pre_path = os.path.join(config.processed_data_dir, "pretrain.pt")
    sft_path = os.path.join(config.processed_data_dir, "sft.pt")
    status.append(f"  {'Pretrain Data:':<18} [{'X' if os.path.exists(pre_path) else ' '}]")
    status.append(f"  {'SFT Data:':<18} [{'X' if os.path.exists(sft_path) else ' '}]")
    
    pretrain_path = config.best_model_path.replace("best.pt", "pretrain_best.pt")
    sft_path = config.best_model_path.replace("best.pt", "sft_best.pt")
    status.append(f"  {'Pretrain Model:':<18} [{'X' if os.path.exists(pretrain_path) else ' '}]")
    status.append(f"  {'SFT Model:':<18} [{'X' if os.path.exists(sft_path) else ' '}]")
    
    return "\n".join(status)
    
def get_validated_input(prompt, default, min_val, max_val, as_int=False):
    while True:
        try:
            val_in = input(f"{prompt} [{default}]: ").strip()
            if not val_in:
                return default
            val = float(val_in)
            if min_val <= val <= max_val:
                return int(val) if as_int else val
            print(f"Error: Value must be between {min_val} and {max_val}.")
        except ValueError:
            print(f"Error: Invalid number. Please enter a {'integer' if as_int else 'float'}.")

def interactive_menu():
    prereqs = {
        "3": [
            (os.path.join(config.tokenizer_dir, "tokenizer.json"), "Tokenizer"),
            (os.path.join(config.processed_data_dir, "pretrain.pt"), "Pretrain Data")
        ],
        "5": [
            (os.path.join(config.tokenizer_dir, "tokenizer.json"), "Tokenizer"),
            (os.path.join(config.processed_data_dir, "sft.pt"), "SFT Data")
        ],
        "8": [
            (os.path.join(config.tokenizer_dir, "tokenizer.json"), "Tokenizer")
        ]
    }
    last_error_choice = None
    while True:
        clear_screen()
        print("  TinyGPT Multi-Tool")
        print("  " + "─" * 30)
        if last_error_choice:
            print(f"  [!] LAST OPERATION FAILED (Option {last_error_choice})")
        print()
        print("  Status")
        print(get_status())
        print()
        print("  Data")
        print("  " + get_data_specs())
        print()
        print("  " + "─" * 30)
        print("  1. Train Tokenizer")
        print("  2. Preprocess Data (Pretrain)")

        req3 = ", ".join(name for path, name in prereqs["3"] if not os.path.exists(path))
        print(f"  3. Pretrain Model" + (f"  [Requires: {req3}]" if req3 else ""))

        print("  4. Preprocess Data (SFT)")

        req5 = ", ".join(name for path, name in prereqs["5"] if not os.path.exists(path))
        print(f"  5. SFT Fine-tune" + (f"  [Requires: {req5}]" if req5 else ""))

        print("  6. Train Image VAE (The Eyes)")
        print("  7. Train Image Diffusion (The Brain)")

        req8 = ", ".join(name for path, name in prereqs["8"] if not os.path.exists(path))
        print(f"  8. Chat" + (f"  [Requires: {req8}]" if req8 else ""))

        print("  9. Exit")
        print("  " + "─" * 30)
        
        choice = input("\n  > ").strip()
        
        if choice in prereqs:
            missing = [name for path, name in prereqs[choice] if not os.path.exists(path)]
            if choice == "8":
                pre_path = config.best_model_path.replace("best.pt", "pretrain_best.pt")
                sft_path = config.best_model_path.replace("best.pt", "sft_best.pt")
                if not os.path.exists(pre_path) and not os.path.exists(sft_path):
                    missing.append("Target Model Checkpoint (Run Option 3 first)")
            
            if missing:
                print(f"\n[!] Error: Missing prerequisites: {', '.join(missing)}")
                input("\nPress Enter to continue...")
                continue

        success = True
        if choice == "1":
            success = run_command(["tokenizer/train_tokenizer.py"])
        elif choice == "2":
            success = run_command(["data/prepare_data.py", "--mode", "pretrain"])
        elif choice == "3":
            success = run_command(["train.py", "--mode", "pretrain"])
        elif choice == "4":
            success = run_command(["data/prepare_data.py", "--mode", "sft"])
        elif choice == "5":
            success = run_command(["train.py", "--mode", "sft"])
        elif choice == "6":
            success = run_command(["-m", "image_generation.train", "--mode", "vae"])
        elif choice == "7":
            success = run_command(["-m", "image_generation.train", "--mode", "diffusion"])
        elif choice == "8":
            print("\n--- Chat Settings ---")
            temp = get_validated_input("Temperature", config.temperature, 0.1, 2.0)
            top_p = get_validated_input("Top-p", config.top_p, 0.0, 1.0)
            max_tokens = get_validated_input("Max New Tokens", config.max_new_tokens, 1, config.max_seq_len, as_int=True)
            success = run_command(["chat.py", "--temperature", str(temp), "--top_p", str(top_p), "--max_new_tokens", str(max_tokens)])
        elif choice == "9":
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please try again.")
            success = False # To not clear error state on invalid input
            
        if success:
            last_error_choice = None
        elif choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            last_error_choice = choice
            
        input("\nPress Enter to continue...")

def main():
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
