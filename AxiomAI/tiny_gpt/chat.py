import os
import sys
import argparse
import torch
import json

import config
from model.transformer import TinyGPT
from tokenizer.bpe import BPETokenizer

def chat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
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
        vocab_size=config.vocab_size,
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
        print("TinyGPT Chat CLI ready. Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("[USER]: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            history.append({"from": "human", "value": user_input})
            
            # Format history for model
            prompt = ""
            for turn in history:
                role_prefix = "[HUMAN]: " if turn["from"] == "human" else "[GPT]: "
                prompt += f"{role_prefix}{turn['value']} "
            
            prompt += "[GPT]: "
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
            
            eos_id = tokenizer.vocab.get("<EOS>", 3)
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=256, 
                temperature=args.temperature, 
                top_p=args.top_p,
                eos_id=eos_id
            )
            
            # Extract only the newly generated part
            new_tokens = output_ids[0][input_ids.size(1):]
            response = tokenizer.decode(new_tokens.tolist())
            
            # Clean up response (BPE might include trailing spaces or EOS markers)
            response = response.replace("<EOS>", "").strip()
            
            if args.json_output:
                print(json.dumps({"response": response, "tokens_generated": len(new_tokens)}))
            else:
                print(f"[GPT]: {response}")
                
            history.append({"from": "gpt", "value": response})
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    chat()
