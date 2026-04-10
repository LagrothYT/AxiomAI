import os
import sys
import argparse
import torch
import json
import math

import config
from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer
from image_generation.bridge import ModelBridge
from utils import safe_load

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    sft_path = config.best_model_path.replace("best.pt", "sft_best.pt")
    pretrain_path = config.best_model_path.replace("best.pt", "pretrain_best.pt")
    model_path = sft_path if os.path.exists(sft_path) else pretrain_path
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found.")
        return
        
    checkpoint = safe_load(model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    bridge = ModelBridge()
    
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
                
            if user_input.strip().startswith("/imagine"):
                bridge_result = bridge.process_chat_message(user_input)
                if bridge_result.get("type") == "image":
                    if not args.json_output:
                        print(f" [SYSTEM] Direct User Routing: Image Pipeline initialized for: {bridge_result['prompt']}")
                    history.append({"from": "human", "value": user_input})
                    history.append({"from": "gpt", "value": "[Image Generated Successfully]"})
                else:
                    if not args.json_output:
                        print(f" [SYSTEM] Error: {bridge_result.get('content', 'Unknown bridge error')}")
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
            
            if not args.json_output:
                print(f"\n {config.gpt_prefix}  > ", end="", flush=True)

            stream = model.generate_stream(
                input_ids, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                top_p=args.top_p,
                eos_id=eos_id
            )
            
            probs = []
            num_tokens = 0
            response = ""
            
            import time
            start_time = time.time()
            initial_seq_len = input_ids.size(1)
            
            for item, val in stream:
                probs.append(val)
                num_tokens += 1
                token_text = tokenizer.decode([item])
                
                if item == eos_id:
                    break
                    
                response += token_text
                if not args.json_output:
                    print(token_text, end="", flush=True)
                
            gen_time = time.time() - start_time
                
            if not args.json_output:
                print()
                
            response = response.strip()
            
            bridge_result = bridge.process_chat_message(response)
            if bridge_result.get("type") == "image":
                if not args.json_output:
                    print(f" [SYSTEM] Routing to Image Generator Pipeline: {bridge_result['prompt']}")
            elif bridge_result.get("type") == "error":
                if not args.json_output:
                    print(f" [SYSTEM] Generator Error: {bridge_result.get('content')}")
            
            # Context info
            context_left = max(0, config.max_seq_len - (initial_seq_len + num_tokens))
            
            # Metrics
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
                print(f" └─ {num_tokens} tok · {tok_per_sec:.1f} tok/s · PPL {perplexity:.2f} · {context_left} ctx left\n")
                
            history.append({"from": "gpt", "value": response})
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    chat()
