import sys
import os
import json
import configparser
import torch
import numpy as np
import math
from trainer import train_model
from chat import start_chat
from preprocess import preprocess_data
from sft_preprocess import preprocess_data as preprocess_sft_data
from tokenizer.my_tokenizer import CharTokenizer

# Allow numpy scalar types in torch.load checkpoints (PyTorch 2.6+ compatibility)
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except (ImportError, AttributeError):
    pass

# Cache for token count to avoid expensive recalculation on every menu refresh
_base_cache = {'tokens': None, 'convos': None, 'mtime': None, 'vocab_mtime': None}
_sft_cache = {'tokens': None, 'convos': None, 'mtime': None, 'vocab_mtime': None}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_config():
    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    return config

def format_param_count(n):
    """Human-readable parameter count: 1.2M, 350K, etc."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

def iter_data_files(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                yield full_path
    else:
        yield path

def get_dataset_stats(cfg, cache_dict):
    if not cfg.has_section('DATA'):
        return False, "❌ CONFIG MISSING [DATA]"
    data_path = cfg['DATA'].get('raw_data_path', '')
    vocab_path = cfg['DATA'].get('vocab_path', 'tokenizer/vocab.json')
    
    if not os.path.exists(data_path):
        return False, "❌ NOT FOUND"
        
    files = [p for p in iter_data_files(data_path) if p.lower().endswith(('.jsonl', '.txt'))]
    size_mb = sum(os.path.getsize(p) for p in files) / (1024 * 1024)
    
    tokenizer = CharTokenizer()
    has_vocab = tokenizer.load(vocab_path)
    
    try:
        dataset_mtime = max([os.path.getmtime(p) for p in files], default=os.path.getmtime(data_path))
        vocab_mtime = os.path.getmtime(vocab_path) if has_vocab else 0
    except OSError:
        dataset_mtime = 0; vocab_mtime = 0
        
    if (cache_dict['mtime'] != dataset_mtime or cache_dict['vocab_mtime'] != vocab_mtime):
        conv_count = 0
        total_tokens = 0 if has_vocab else "?"
        try:
            for file_path in files:
                if file_path.lower().endswith('.jsonl'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                conv_count += 1
                                if has_vocab:
                                    try:
                                        data = json.loads(line)
                                        for msg in data.get('conversations', []):
                                            role = msg.get('from', 'unknown')
                                            val = msg.get('value', '')
                                            total_tokens += len(tokenizer.encode(f"<{role}>: {val}\n"))
                                            if role == 'gpt': total_tokens += 1
                                        total_tokens += 1
                                    except: pass
                elif has_vocab:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        conv_count += 1
                        total_tokens += len(tokenizer.encode(text)) + 1
        except:
            conv_count = "?"
            
        cache_dict['convos'] = conv_count
        cache_dict['tokens'] = total_tokens
        cache_dict['mtime'] = dataset_mtime
        cache_dict['vocab_mtime'] = vocab_mtime
        
    tokens_str = f"{cache_dict['tokens']:,}" if isinstance(cache_dict['tokens'], int) else "?"
    conv_str = f"{cache_dict['convos']:,}" if isinstance(cache_dict['convos'], int) else "?"
    
    return True, f"Size: {size_mb:.2f}MB │ Convos: {conv_str} │ Tokens: {tokens_str}"

def get_dashboard_status(cfg, sft_cfg):
    lines = []
    sep = "─" * 60
    lines.append("╔══════════════════════════════════════════════════════════╗")
    lines.append("║                 A X I O M   A I   v1.0                   ║")
    lines.append("╚══════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("  [ PIPELINE STATUS ]")
    lines.append(sep)
    
    # --- BASE PIPELINE ---
    lines.append("  🟦 BASE PRE-TRAIN")
    has_base, base_dt_str = get_dataset_stats(cfg, _base_cache)
    lines.append(f"     Data  : {base_dt_str}")
    
    train_path = cfg['DATA'].get('train_path', '')
    if os.path.exists(train_path):
        lines.append(f"     Arrays: ✅ READY")
    else:
        lines.append("     Arrays: ❌ NOT READY")
        
    model_path = cfg['TRAINING'].get('checkpoint_path', '')
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            best_l = ckpt.get('best_val_loss', float('inf'))
            params = sum(t.numel() for t in ckpt['model'].values()) if 'model' in ckpt else 0
            if best_l != float('inf'):
                ppl = math.exp(min(best_l, 20.0))
                lines.append(f"     Model : ✅ TRAINED ({format_param_count(params)}) │ PPL: {ppl:.2f}")
            else:
                lines.append(f"     Model : ✅ CHECKPOINT ({format_param_count(params)})")
        except:
            lines.append("     Model : ✅ CHECKPOINT FOUND")
    else:
        lines.append("     Model : ❌ NOT TRAINED")

    lines.append("")
    # --- SFT PIPELINE ---
    lines.append("  🟧 SUPERVISED FINE-TUNE")
    has_sft, sft_dt_str = get_dataset_stats(sft_cfg, _sft_cache)
    lines.append(f"     Data  : {sft_dt_str}")
    
    sft_train_path = sft_cfg['DATA'].get('train_path', '')
    if os.path.exists(sft_train_path):
        lines.append(f"     Arrays: ✅ READY")
    else:
        lines.append("     Arrays: ❌ NOT READY")
        
    sft_model_path = sft_cfg['TRAINING'].get('checkpoint_path', '')
    if os.path.exists(sft_model_path):
        try:
            ckpt = torch.load(sft_model_path, map_location='cpu', weights_only=False)
            best_l = ckpt.get('best_val_loss', float('inf'))
            params = sum(t.numel() for t in ckpt['model'].values()) if 'model' in ckpt else 0
            if best_l != float('inf'):
                ppl = math.exp(min(best_l, 20.0))
                lines.append(f"     Model : ✅ TRAINED ({format_param_count(params)}) │ PPL: {ppl:.2f}")
            else:
                lines.append(f"     Model : ✅ CHECKPOINT ({format_param_count(params)})")
        except:
            lines.append("     Model : ✅ CHECKPOINT FOUND")
    else:
        lines.append("     Model : ❌ NOT TRAINED")

    # --- VIDEO PIPELINE ---
    lines.append("")
    lines.append("  🎥 VIDEO GENERATION")
    v_cfg = configparser.ConfigParser()
    if os.path.exists('configs/video_config.ini'):
        v_cfg.read('configs/video_config.ini')
        v_data = v_cfg['DATA'].get('video_data_path', 'data/Videos/')
        lines.append(f"     Data  : {v_data}")
        
        v_train = v_cfg['TRAINING']
        v_data_cfg = v_cfg['DATA']
        v_w = int(v_train.get('width', 64))
        v_h = int(v_train.get('height', 64))
        v_fps = float(v_data_cfg.get('fps', 8.0))
        v_duration = float(v_data_cfg.get('duration', 2.0))
        v_frames = int(v_fps * v_duration)
        
        # Count actual video files in the data directory
        v_path = v_data_cfg.get('video_data_path', 'data/Videos/')
        if os.path.isdir(v_path):
            v_count = len([f for f in os.listdir(v_path) if f.endswith('.mp4')])
        else:
            v_count = 0
        
        lines.append(f"     Config: ✅ READY [ {v_w}x{v_h} | {v_duration}s @ {v_fps:.0f}fps = {v_frames} frames ]")
        lines.append(f"     Videos: {v_count} .mp4 files found")
        
        v_model_path = v_train.get('checkpoint_path', '')
        if os.path.exists(v_model_path):
            try:
                ckpt = torch.load(v_model_path, map_location='cpu', weights_only=False)
                best_l = ckpt.get('best_val_loss', float('inf'))
                params = 0
                if 'video_model' in ckpt:
                    params += sum(t.numel() for t in ckpt['video_model'].values())
                if 'text_encoder' in ckpt:
                    params += sum(t.numel() for t in ckpt['text_encoder'].values())
                
                if best_l != float('inf'):
                    ppl = math.exp(min(best_l, 20.0))
                    lines.append(f"     Model : ✅ TRAINED ({format_param_count(params)}) │ PPL: {ppl:.2f}")
                else:
                    lines.append(f"     Model : ✅ CHECKPOINT ({format_param_count(params)})")
            except:
                lines.append("     Model : ✅ CHECKPOINT FOUND")
        else:
            lines.append("     Model : ❌ NOT TRAINED")
    else:
        lines.append("     Config: ❌ MISSING")

    lines.append(sep)
    
    # --- GLOBAL ---
    tok_path = cfg['DATA'].get('vocab_path', '')
    if os.path.exists(tok_path):
        lines.append("  🔤 Shared Tokenizer : ✅ TRAINED")
    else:
        lines.append("  🔤 Shared Tokenizer : ❌ NOT TRAINED")
        
    d = cfg['MODEL'].get('d_model', '?')
    l = cfg['MODEL'].get('n_layers', '?')
    h = cfg['MODEL'].get('n_heads', '?')
    dev = cfg['TRAINING'].get('device', 'auto').upper()
    lines.append(f"  ⚙  Hardware Target  : {dev} [ {l}L {h}H d={d} ]")

    return "\n".join(lines)

def main_menu():
    cfg = load_config()
    sft_cfg = configparser.ConfigParser()
    if os.path.exists('configs/sft_config.ini'):
        sft_cfg.read('configs/sft_config.ini')
        
    clear_screen()

    while True:
        print(get_dashboard_status(cfg, sft_cfg))
        print()
        print("┌──────────────────────────────────────────────┐")
        print("│  [ GLOBAL & DATA PREPARATION ]               │")
        print("│  1 │ Train Shared Tokenizer                  │")
        print("│  2 │ Parse PRETRAIN Data (.jsonl)            │")
        print("│  3 │ Parse FINE-TUNE Data (.jsonl)           │")
        print("│                                              │")
        print("│  [ TEXT AI PIPELINE ]                        │")
        print("│  4 │ Train BASE Text Model                   │")
        print("│  5 │ Train SFT Text Model                    │")
        print("│  6 │ Chat with Model                         │")
        print("│                                              │")
        print("│  [ VIDEO AI PIPELINE ]                       │")
        print("│  7 │ Train Video VAE (Pixels -> Tokens)      │")
        print("│  8 │ Pre-Train Text Encoder (Language)       │")
        print("│  9 │ Train Video Model (ST-Transformer)      │")
        print("│                                              │")
        print("│  [ EXPORT ]                                  │")
        print("│ 10 │ Export Models to GGUF                   │")
        print("│                                              │")
        print("│  0 │ Exit                                    │")
        print("└──────────────────────────────────────────────┘")

        try:
            choice = input("\n  Select (0-10): ").strip()

            if choice == '1':
                tokenizer = CharTokenizer()
                vocab_path = cfg['DATA']['vocab_path']
                raw_data = cfg['DATA']['raw_data_path']
                vocab_size = int(cfg['MODEL']['vocab_size'])

                print("\n-- Training Tokenizer --")
                tokenizer.train_from_file(raw_data, vocab_size, vocab_path)
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '2':
                preprocess_data()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '3':
                preprocess_sft_data()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '4':
                train_model()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '5':
                train_model(is_sft=True)
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '6':
                start_chat()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '7':
                from video_trainer import train_vae
                train_vae()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '8':
                from video_trainer import train_text_encoder
                train_text_encoder()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '9':
                from video_trainer import train_video_model
                train_video_model()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '10':
                from export_gguf import export_menu
                export_menu()
                input("\nPress Enter to continue...")
                clear_screen()

            elif choice == '0':
                print("Exiting AXIOM AI. Goodbye!")
                sys.exit()

            else:
                print("Invalid selection. Please try again.")
                input("\nPress Enter to continue...")
                clear_screen()

        except KeyboardInterrupt:
            print("\n  [System] Ctrl+C Detected. Exiting AXIOM AI. Goodbye!")
            sys.exit()

if __name__ == "__main__":
    main_menu()
