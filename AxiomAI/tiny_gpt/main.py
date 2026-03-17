import argparse
import sys
import os
import subprocess

def run_command(command):
    try:
        # We use sys.executable to ensure we use the same python interpreter
        full_command = [sys.executable] + command
        print(f"Running: {' '.join(full_command)}")
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def interactive_menu():
    while True:
        print("\n=== TinyGPT Multi-Tool ===")
        print("1. Train Tokenizer")
        print("2. Data Preprocessing (Pretrain mode)")
        print("3. Pretrain Model")
        print("4. Data Preprocessing (SFT mode)")
        print("5. SFT Fine-tune Model")
        print("6. Interactive Chat")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == "1":
            run_command(["tokenizer/train_tokenizer.py"])
        elif choice == "2":
            run_command(["data/prepare_data.py", "--mode", "pretrain"])
        elif choice == "3":
            run_command(["train.py", "--mode", "pretrain"])
        elif choice == "4":
            run_command(["data/prepare_data.py", "--mode", "sft"])
        elif choice == "5":
            run_command(["train.py", "--mode", "sft"])
        elif choice == "6":
            run_command(["chat.py"])
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please try again.")

def main():
    if len(sys.argv) > 1:
        # Keep argument-based CLI for automation/scripts
        parser = argparse.ArgumentParser(description="TinyGPT Unified CLI")
        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # Train Tokenizer
        subparsers.add_parser("train-tokenizer", help="Train the BPE tokenizer on data/*.jsonl")

        # Prepare Data
        prep_parser = subparsers.add_parser("prepare", help="Preprocess data for training")
        prep_parser.add_argument("--mode", choices=["pretrain", "sft"], required=True, help="Preprocessing mode")

        # Train Model
        train_parser = subparsers.add_parser("train", help="Train the model (Pretrain or SFT)")
        train_parser.add_argument("--mode", choices=["pretrain", "sft"], required=True, help="Training mode")

        # Chat
        chat_parser = subparsers.add_parser("chat", help="Chat with the trained model")
        chat_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
        chat_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")

        args = parser.parse_args()

        if args.command == "train-tokenizer":
            run_command(["tokenizer/train_tokenizer.py"])
        elif args.command == "prepare":
            run_command(["data/prepare_data.py", "--mode", args.mode])
        elif args.command == "train":
            run_command(["train.py", "--mode", args.mode])
        elif args.command == "chat":
            cmd = ["chat.py", "--temperature", str(args.temperature), "--top_p", str(args.top_p)]
            run_command(cmd)
        else:
            parser.print_help()
    else:
        # Default to interactive menu if no arguments
        interactive_menu()

if __name__ == "__main__":
    main()
