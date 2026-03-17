import sys
import os
import subprocess
import config

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
    
    # Tokenizer
    tok_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    tok_exists = os.path.exists(tok_path)
    status.append(f"{'Tokenizer:':<18} [{'X' if tok_exists else ' '}]")
    
    # Data
    pre_path = os.path.join(config.processed_data_dir, "pretrain.pt")
    sft_path = os.path.join(config.processed_data_dir, "sft.pt")
    status.append(f"{'Pretrain Data:':<18} [{'X' if os.path.exists(pre_path) else ' '}]")
    status.append(f"{'SFT Data:':<18} [{'X' if os.path.exists(sft_path) else ' '}]")
    
    # Model
    model_exists = os.path.exists(config.best_model_path)
    status.append(f"{'Model Checkpoint:':<18} [{'X' if model_exists else ' '}]")
    
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
        "6": [
            (os.path.join(config.tokenizer_dir, "tokenizer.json"), "Tokenizer"),
            (config.best_model_path, "Model Checkpoint")
        ]
    }
    last_error_choice = None
    while True:
        clear_screen()
        print("=== TinyGPT Multi-Tool ===")
        if last_error_choice:
            print(f" [!] LAST OPERATION FAILED (Option {last_error_choice})")
        print(get_status())
        print("-" * 26)
        print("1. Train Tokenizer")
        print("2. Data Preprocessing (Pretrain mode)")
        
        req3 = ", ".join(name for path, name in prereqs["3"] if not os.path.exists(path))
        print(f"3. Pretrain Model {'[Requires: ' + req3 + ']' if req3 else ''}")
        
        print("4. Data Preprocessing (SFT mode)")
        
        req5 = ", ".join(name for path, name in prereqs["5"] if not os.path.exists(path))
        print(f"5. SFT Fine-tune Model {'[Requires: ' + req5 + ']' if req5 else ''}")
        
        req6 = ", ".join(name for path, name in prereqs["6"] if not os.path.exists(path))
        print(f"6. Interactive Chat {'[Requires: ' + req6 + ']' if req6 else ''}")
        
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice in prereqs:
            missing = [name for path, name in prereqs[choice] if not os.path.exists(path)]
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
            print("\n--- Chat Settings ---")
            temp = get_validated_input("Temperature", config.temperature, 0.1, 2.0)
            top_p = get_validated_input("Top-p", config.top_p, 0.0, 1.0)
            max_tokens = get_validated_input("Max New Tokens", config.max_new_tokens, 1, config.max_seq_len, as_int=True)
            success = run_command(["chat.py", "--temperature", str(temp), "--top_p", str(top_p), "--max_new_tokens", str(max_tokens)])
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please try again.")
            success = False # To not clear error state on invalid input
            
        if success:
            last_error_choice = None
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            last_error_choice = choice
            
        input("\nPress Enter to continue...")

def main():
    interactive_menu()

if __name__ == "__main__":
    main()
