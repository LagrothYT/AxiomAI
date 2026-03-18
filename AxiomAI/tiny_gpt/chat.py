import os
import sys
import argparse
import torch
import json
import math

import config
from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "╔" + "═"*48 + "╗")
    print("║" + " TinyGPT Interactive Chat ".center(48) + "║")
    print("║" + " Commands: /quit, /clear, /help ".center(48) + "║")
    print("╚" + "═"*48 + "╝\n")

def chat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=config.temperature)
    parser.add_argument("--top_p", type=float, default=config.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=config.max_new_tokens)
    parser.add_argument("--json_output", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # Load tokenizer
    tokenizer_path = os.path.join(config.tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Load model
    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=0.0 # No dropout during inference
    ).to(device)
    
    if not os.path.exists(config.best_model_path):
        print(f"Error: Model checkpoint not found at {config.best_model_path}")
        return
        
    model.load_state_dict(torch.load(config.best_model_path, map_location=device, weights_only=True))
    model.eval()
    
    history = []
    
    if not args.json_output:
        print_header()
    
    while True:
        try:
            user_input = input(f" {config.user_prefix} > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "/quit":
                break
            if user_input.lower() == "/clear":
                history = []
                if not args.json_output:
                    print_header()
                    print("--- Chat Cleared ---\n")
                continue
                
            if user_input.lower() == "/help":
                if not args.json_output:
                    print("\n--- Available Commands ---")
                    print(" /quit  - Exit the chat")
                    print(" /clear - Clear chat history and screen")
                    print(" /help  - Show this help message")
                    print("\n--- Current Settings ---")
                    print(f" Temperature: {args.temperature}")
                    print(f" Top-p:       {args.top_p}")
                    print(f" Max Context: {config.max_seq_len} tokens")
                    print("-" * 26 + "\n")
                continue
                
            history.append({"from": "human", "value": user_input})
            
            # Format history and trim if it exceeds context limit
            trimmed = False
            truncated = False
            while True:
                prompt = ""
                for turn in history:
                    role_prefix = f"{config.human_role}: " if turn["from"] == "human" else f"{config.gpt_role}: "
                    prompt += f"{role_prefix}{turn['value']} "
                prompt += f"{config.gpt_role}: "
                
                encoded_prompt = tokenizer.encode(prompt)
                
                # If it fits, we're done
                if len(encoded_prompt) + args.max_new_tokens <= config.max_seq_len:
                    break
                
                # If we have history to pop, pop it
                if len(history) > 1:
                    history.pop(0)
                    # Ensure the conversation doesn't start with a bot response
                    if history and history[0]["from"] == "gpt":
                        history.pop(0)
                    trimmed = True
                else:
                    # Single message still too long: truncate from front
                    limit = config.max_seq_len - args.max_new_tokens
                    encoded_prompt = encoded_prompt[-limit:]
                    truncated = True
                    break
            
            if not args.json_output:
                if truncated:
                    print(f" [SYSTEM] Warning: Message significantly exceeds limit. Truncated to {config.max_seq_len - args.max_new_tokens} tokens.")
                elif trimmed:
                    print(" [SYSTEM] Warning: Old context trimmed to fit limit.")

            input_ids = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
            
            # Context info (calculated after generation below)
            
            eos_id = tokenizer.vocab.get(config.eos_token, 3)
            # generate now returns (ids, probs, time)
            output_ids, probs, gen_time = model.generate(
                input_ids, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                top_p=args.top_p,
                eos_id=eos_id
            )
            
            # Context info
            context_left = config.max_seq_len - output_ids.size(1)
            
            # Extract only the newly generated part
            new_tokens = output_ids[0][input_ids.size(1):]
            response = tokenizer.decode(new_tokens.tolist())
            response = response.replace(config.eos_token, "").strip()
            
            # Metrics
            num_tokens = new_tokens.size(0)
            # Perplexity = exp(-mean(log(probs)))
            perplexity = math.exp(-sum(math.log(max(p, 1e-10)) for p in probs) / len(probs)) if probs else 1.0
            tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0
            
            if args.json_output:
                print(json.dumps({
                    "response": response, 
                    "tokens": num_tokens,
                    "perplexity": f"{perplexity:.2f}",
                    "tok_s": f"{tok_per_sec:.1f}",
                    "context_left": context_left
                }))
            else:
                print(f"\n {config.gpt_prefix}  > {response}")
                print(f" └─ METRICS: [Perplexity: {perplexity:.2f}] [Tokens: {num_tokens}] [Speed: {tok_per_sec:.1f} tok/s] [Context Left: {context_left}]")
                print("-" * 50)
                
            history.append({"from": "gpt", "value": response})
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    chat()
