import torch
import configparser
import os
import sys
import math
import time
from model import Transformer
from tokenizer.my_tokenizer import CharTokenizer
import torch.nn.functional as F

# Allow numpy scalar types in torch.load checkpoints (PyTorch 2.6+ compatibility)
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except (ImportError, AttributeError):
    pass

# Optimize CPU matrix multiplication precision
torch.set_float32_matmul_precision('high')

def top_k_filter(logits, k=40):
    """Zero out all logits outside the top-k, then return filtered logits."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    top_vals, _ = torch.topk(logits, k)
    threshold = top_vals[-1]
    logits[logits < threshold] = float('-inf')
    return logits

def top_p_filter(logits, p=0.9):
    if p >= 1.0 or p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter back to the original index positions
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def apply_repetition_penalty(logits, history_tokens, penalty=1.15):
    if penalty <= 1.0 or not history_tokens:
        return logits
    
    unique_tokens = list(set(history_tokens))
    token_tensor = torch.tensor(unique_tokens, dtype=torch.long, device=logits.device)
    
    # Apply penalty. For positive logits, divide. For negative logits, multiply.
    score = torch.gather(logits, -1, token_tensor)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(-1, token_tensor, score)
    return logits

def safe_set_torch_threads(num_threads):
    if num_threads <= 0:
        return
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except RuntimeError:
        pass

def select_chat_checkpoint(config):
    base_model_path = config['TRAINING']['checkpoint_path']
    sft_config = configparser.ConfigParser()
    sft_config.read('configs/sft_config.ini')
    sft_model_path = sft_config['TRAINING'].get('checkpoint_path', 'model/sft_best_model.pth') if sft_config.has_section('TRAINING') else 'model/sft_best_model.pth'

    if os.path.exists(sft_model_path):
        return sft_model_path, "SFT Fine-Tuned", sft_config if sft_config.has_section('CHAT') else config
    if os.path.exists(base_model_path):
        return base_model_path, "Base Pre-Trained", config
    raise FileNotFoundError(f"Model weights not found at {base_model_path}. Train a text model first.")

def sample_next_token(logits, tokenizer, prompt_tokens, generated_tokens, temperature, top_k, top_p, rep_pen):
    raw_logits = logits[0, -1, :]
    log_probs = F.log_softmax(raw_logits, dim=-1)
    next_token_logits = raw_logits / temperature
    next_token_logits = apply_repetition_penalty(next_token_logits, prompt_tokens + generated_tokens, penalty=rep_pen)
    next_token_logits = top_k_filter(next_token_logits, k=top_k)
    next_token_logits = top_p_filter(next_token_logits, p=top_p)
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token, -log_probs[next_token].item()

class AxiomChatEngine:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('configs/config.ini')
        self.device = "cpu"

        num_threads = int(self.config['TRAINING'].get('num_threads', 0))
        safe_set_torch_threads(num_threads)

        self.tokenizer = CharTokenizer()
        if not self.tokenizer.load(self.config['DATA']['vocab_path']):
            raise FileNotFoundError("Tokenizer not found. Train it first.")

        self.model_path, self.model_source, self.chat_cfg = select_chat_checkpoint(self.config)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model_cfg = checkpoint.get('config', dict(self.config['MODEL']))
        self.max_seq_len = int(self.model_cfg.get('max_seq_len', self.config['MODEL'].get('max_seq_len', '256')))

        self.model = Transformer(self.model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def default_settings(self):
        chat_cfg = self.chat_cfg
        return {
            "temperature": float(chat_cfg['CHAT'].get('temperature', '0.8')) if chat_cfg.has_section('CHAT') else 0.8,
            "top_k": int(chat_cfg['CHAT'].get('top_k', '40')) if chat_cfg.has_section('CHAT') else 40,
            "top_p": float(chat_cfg['CHAT'].get('top_p', '0.9')) if chat_cfg.has_section('CHAT') else 0.9,
            "repetition_penalty": float(chat_cfg['CHAT'].get('repetition_penalty', '1.15')) if chat_cfg.has_section('CHAT') else 1.15,
            "max_gen_length": int(chat_cfg['CHAT'].get('max_gen_length', '200')) if chat_cfg.has_section('CHAT') else 200,
        }

    def status(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "source": self.model_source,
            "path": self.model_path,
            "params": num_params,
            "max_seq_len": self.max_seq_len,
        }

    def generate(self, history, settings=None):
        cfg = self.default_settings()
        if settings:
            cfg.update({k: v for k, v in settings.items() if v is not None})

        temperature = max(0.01, min(float(cfg["temperature"]), 2.0))
        top_k = max(0, int(cfg["top_k"]))
        top_p = max(0.0, min(float(cfg["top_p"]), 1.0))
        rep_pen = max(1.0, float(cfg["repetition_penalty"]))
        max_gen_length = max(1, int(cfg["max_gen_length"]))

        turns = []
        for item in history or []:
            role = item.get("role", "human")
            value = str(item.get("value", "")).strip()
            if role in ("human", "gpt", "system") and value:
                turns.append({"role": role, "value": value})

        max_prompt_len = self.max_seq_len - min(max_gen_length, self.max_seq_len // 2) - 1
        max_prompt_len = max(1, max_prompt_len)

        while True:
            full_context = ""
            for turn in turns:
                full_context += f"<{turn['role']}>: {turn['value']}\n"
            full_context += "<gpt>:"

            prompt_tokens = self.tokenizer.encode(full_context)
            if len(prompt_tokens) > max_prompt_len and len(turns) > 1:
                turns.pop(0)
            else:
                break

        if not prompt_tokens:
            return {"response": "", "ppl": 0.0, "context_left": self.max_seq_len}
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[-max_prompt_len:]
        prompt_tokens = [self.tokenizer.bos_id] + prompt_tokens

        self.model.reset_cache()
        generated_tokens = []
        nll_sum = 0.0
        total_len = len(prompt_tokens)

        with torch.no_grad():
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            logits, _ = self.model(prompt_tensor, use_cache=True)
            next_token, nll = sample_next_token(logits, self.tokenizer, prompt_tokens, generated_tokens, temperature, top_k, top_p, rep_pen)

            if next_token != self.tokenizer.eos_id:
                generated_tokens.append(next_token)
                nll_sum += nll
                input_id = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                total_len += 1

                for _ in range(max_gen_length - 1):
                    if total_len >= self.max_seq_len:
                        break
                    logits, _ = self.model(input_id, use_cache=True)
                    next_token, nll = sample_next_token(logits, self.tokenizer, prompt_tokens, generated_tokens, temperature, top_k, top_p, rep_pen)
                    if next_token == self.tokenizer.eos_id:
                        break
                    generated_tokens.append(next_token)
                    nll_sum += nll
                    total_len += 1
                    input_id[0, 0] = next_token

        response = self.tokenizer.decode(generated_tokens).strip()
        ppl = math.exp(min(nll_sum / len(generated_tokens), 20.0)) if generated_tokens else 0.0
        return {
            "response": response,
            "ppl": ppl,
            "context_left": max(0, self.max_seq_len - total_len),
            "source": self.model_source,
        }


def start_chat():
    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    device = "cpu"
    
    num_threads = int(config['TRAINING'].get('num_threads', 0))
    safe_set_torch_threads(num_threads)

    tokenizer = CharTokenizer()
    if not tokenizer.load(config['DATA']['vocab_path']):
        print("Error: Tokenizer not found. Train it first.")
        return

    base_model_path = config['TRAINING']['checkpoint_path']
    
    sft_config = configparser.ConfigParser()
    sft_config.read('configs/sft_config.ini')
    sft_model_path = sft_config['TRAINING'].get('checkpoint_path', 'model/sft_best_model.pth')
    
    if os.path.exists(sft_model_path):
        model_path = sft_model_path
        model_source = "SFT Fine-Tuned"
        chat_cfg = sft_config if sft_config.has_section('CHAT') else config
    elif os.path.exists(base_model_path):
        model_path = base_model_path
        model_source = "Base Pre-Trained"
        chat_cfg = config
    else:
        print(f"Error: Model weights not found at {base_model_path}. Train it first.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', dict(config['MODEL']))

    # Read generation hyperparameters from config (with sane defaults)
    temperature = float(chat_cfg['CHAT'].get('temperature', '0.8')) if chat_cfg.has_section('CHAT') else 0.8
    top_k = int(chat_cfg['CHAT'].get('top_k', '40')) if chat_cfg.has_section('CHAT') else 40
    top_p = float(chat_cfg['CHAT'].get('top_p', '0.9')) if chat_cfg.has_section('CHAT') else 0.9
    rep_pen = float(chat_cfg['CHAT'].get('repetition_penalty', '1.15')) if chat_cfg.has_section('CHAT') else 1.15
    max_gen_length = int(chat_cfg['CHAT'].get('max_gen_length', '200')) if chat_cfg.has_section('CHAT') else 200
    max_seq_len = int(model_cfg.get('max_seq_len', config['MODEL'].get('max_seq_len', '256')))

    model = Transformer(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    d_model = model_cfg.get('d_model', '?')
    n_layers = model_cfg.get('n_layers', '?')
    n_heads = model_cfg.get('n_heads', '?')

    def format_param_count(n):
        if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000: return f"{n / 1_000:.1f}K"
        return str(n)

    # ANSI Colors
    RESET = '\033[0m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    DIM = '\033[2m'
    YELLOW = '\033[93m'

    try:
        import psutil
    except ImportError:
        psutil = None
    
    initial_ram = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0.0

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{CYAN}╔══════════════════════════════════════════════════════╗")
    print(f"║               A X I O M   A I   C H A T              ║")
    print(f"╚══════════════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  🧠 Model       {format_param_count(num_params)} params  │  {n_layers}L {n_heads}H d={d_model}")
    print(f"  📦 Weights     {model_source}  │  {os.path.basename(model_path)}")
    print(f"  ⚙  Config      {device.upper()}  │  Seq: {max_seq_len}  │  Max Gen: {max_gen_length}")
    print(f"  🎛  Sampling    Temp: {temperature:.2f}  │  Top-K: {top_k}  │  Top-P: {top_p}  │  Rep: {rep_pen}")
    print(f"  💾 Memory      RAM: {initial_ram:.0f} MB")
    print(f"{DIM}  Commands: 'exit'  │  '/temp <val>'  │  '/video <prompt>'{RESET}")
    print("─" * 56)

    conversation_history = []

    while True:
        try:
            print(f"{GREEN}┌─[ You ]{RESET}")
            user_input = input(f"{GREEN}│ {RESET}").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                break

            # Dynamic temperature command
            if user_input.lower().startswith('/temp '):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.01, min(temperature, 2.0))
                    print(f"{YELLOW}  [System] Temperature set to {temperature:.2f}{RESET}\n")
                except (ValueError, IndexError):
                    print(f"{YELLOW}  [System] Usage: /temp <0.01 - 2.0>{RESET}\n")
                continue

            # Video generation command
            if user_input.lower().startswith('/video '):
                prompt = user_input[7:].strip()
                if prompt:
                    try:
                        from video_generation.generate import generate_video
                        def format_eta(seconds):
                            if seconds is None or seconds == float('inf'):
                                return "--:--"
                            seconds = max(0, int(seconds))
                            mins, secs = divmod(seconds, 60)
                            return f"{mins:02d}:{secs:02d}"

                        progress_line = {"len": 0}

                        def show_video_progress(phase, detail="", current=None, total=None, elapsed=None):
                            bar_width = 24
                            if current is None or total is None or total <= 0:
                                text = f"  Video [{'=' * 3}{'.' * (bar_width - 3)}] preparing...        ETA --:--"
                            else:
                                pct = max(0.0, min(1.0, current / total))
                                filled = min(bar_width, int(round(pct * bar_width)))
                                bar = ("=" * filled) + ("." * (bar_width - filled))
                                if elapsed is not None and current > 0:
                                    eta = elapsed * max(0, total - current) / current
                                else:
                                    eta = None
                                text = f"  Video [{bar}] {pct*100:5.1f}%  {phase:<7} ETA {format_eta(eta)}"

                            clear = " " * max(0, progress_line["len"] - len(text))
                            sys.stdout.write(f"\r{YELLOW}{text}{clear}{RESET}")
                            sys.stdout.flush()
                            progress_line["len"] = len(text)

                        out_path = generate_video(prompt, "chat_gen.mp4", progress_callback=show_video_progress)
                        sys.stdout.write("\n")
                        print(f"{GREEN}  Video saved: {os.path.abspath(out_path)}{RESET}\n", flush=True)
                    except Exception as e:
                        sys.stdout.write("\n")
                        print(f"{YELLOW}  [Error] Video Generation failed: {e}{RESET}\n")
                else:
                    print(f"{YELLOW}  [System] Usage: /video <prompt>{RESET}\n")
                continue

            conversation_history.append({'role': 'human', 'value': user_input})

            # Ensure history length fits into sequence limit. 
            # Slice oldest entire messages instead of randomly truncating the token sequence mid-conversation.
            max_prompt_len = max_seq_len - min(max_gen_length, max_seq_len // 2) - 1
            max_prompt_len = max(1, max_prompt_len)
            
            while True:
                full_context = ""
                for turn in conversation_history:
                    full_context += f"<{turn['role']}>: {turn['value']}\n"
                full_context += "<gpt>:"

                prompt_tokens = tokenizer.encode(full_context)
                if not prompt_tokens:
                    break
                    
                if len(prompt_tokens) > max_prompt_len and len(conversation_history) > 1:
                    # Pop the oldest turn to make space
                    conversation_history.pop(0)
                else:
                    break
            
            if not prompt_tokens:
                continue
            if len(prompt_tokens) > max_prompt_len:
                prompt_tokens = prompt_tokens[-max_prompt_len:]

            prompt_tokens = [tokenizer.bos_id] + prompt_tokens

            model.reset_cache()
            generated_tokens = []
            nll_sum = 0.0

            print(f"{MAGENTA}├─[ Axiom ]{RESET}")
            sys.stdout.write(f"{MAGENTA}│ {RESET}")
            sys.stdout.flush()

            with torch.no_grad():
                # --- Prefill phase ---
                prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                logits, _ = model(prompt_tensor, use_cache=True)

                raw_logits = logits[0, -1, :]
                
                # Calculate negative log-likelihood (confidence mapped to perplexity)
                log_probs = F.log_softmax(raw_logits, dim=-1)
                
                # Sample next token
                next_token_logits = raw_logits / temperature
                next_token_logits = apply_repetition_penalty(next_token_logits, prompt_tokens + generated_tokens, penalty=rep_pen)
                next_token_logits = top_k_filter(next_token_logits, k=top_k)
                next_token_logits = top_p_filter(next_token_logits, p=top_p)
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                nll_sum += -log_probs[next_token].item()

                if next_token != tokenizer.eos_id:
                    generated_tokens.append(next_token)
                    sys.stdout.write(f"{MAGENTA}{tokenizer.decode([next_token])}{RESET}")
                    sys.stdout.flush()

                    # --- Decode phase ---
                    input_id = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    total_len = len(prompt_tokens) + 1  # track total sequence length
                    last_mem_time = time.perf_counter()

                    for _ in range(max_gen_length - 1):
                        if total_len >= max_seq_len:
                            break  # no room left in the RoPE/context window

                        # Dynamic 1.0s Window Title Updating
                        if time.perf_counter() - last_mem_time > 1.0:
                            ram_mb = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0.0
                            last_mem_time = time.perf_counter()
                            
                            title = f"Axiom AI Chat  |  RAM: {ram_mb:.0f} MB"
                            if os.name == 'nt':
                                import ctypes
                                ctypes.windll.kernel32.SetConsoleTitleW(title)
                            else:
                                sys.stdout.write(f"\033]0;{title}\007")
                                sys.stdout.flush()
                        logits, _ = model(input_id, use_cache=True)
                        raw_logits = logits[0, -1, :]
                        
                        log_probs = F.log_softmax(raw_logits, dim=-1)
                        
                        next_token_logits = raw_logits / temperature
                        next_token_logits = apply_repetition_penalty(next_token_logits, prompt_tokens + generated_tokens, penalty=rep_pen)
                        next_token_logits = top_k_filter(next_token_logits, k=top_k)
                        next_token_logits = top_p_filter(next_token_logits, p=top_p)
                        
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()

                        if next_token == tokenizer.eos_id:
                            break

                        nll_sum += -log_probs[next_token].item()
                        generated_tokens.append(next_token)
                        total_len += 1

                        sys.stdout.write(f"{MAGENTA}{tokenizer.decode([next_token])}{RESET}")
                        sys.stdout.flush()

                        input_id[0, 0] = next_token
                else:
                    total_len = len(prompt_tokens)

            print()  # Finalize stream row
            
            # Save the finalized AI response to history
            ai_response = tokenizer.decode(generated_tokens).strip()
            conversation_history.append({'role': 'gpt', 'value': ai_response})

            # Calculate Perplexity for this specific generation
            if len(generated_tokens) > 0:
                avg_nll = nll_sum / len(generated_tokens)
                generation_ppl = math.exp(min(avg_nll, 20.0))
            else:
                generation_ppl = 0.0
                
            context_left = max_seq_len - total_len
            
            # Post-Generation Memory Update
            final_ram = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0.0
            
            # Footer UI
            print(f"{DIM}└─ [ 🧠 PPL: {generation_ppl:.1f}  │  📏 Context Left: {context_left}/{max_seq_len}  │  RAM: {final_ram:.0f}M ]{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}  [System] Interrupted by user.{RESET}\n")
            continue
