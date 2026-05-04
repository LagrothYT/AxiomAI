import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import configparser
import os
import sys
import math
import json
import time
from tqdm import tqdm
from model import Transformer
from tokenizer.my_tokenizer import CharTokenizer
from overfit_monitor import OverfitMonitor
from axiom_ui import print_run_banner

STRUCTURAL_MODEL_KEYS = {
    'd_model',
    'n_layers',
    'n_heads',
    'vocab_size',
    'max_seq_len',
    'hidden_mult',
    'norm_eps',
    'use_moe',
    'num_experts',
    'experts_per_token',
}

# Allow numpy scalar types in torch.load checkpoints (PyTorch 2.6+ compatibility)
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except (ImportError, AttributeError):
    pass

# Optimize CPU matrix multiplication precision
torch.set_float32_matmul_precision('high')
try:
    torch.set_flush_denormal(True)
except (AttributeError, RuntimeError):
    pass
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = True

class TokenDataset(Dataset):
    def __init__(self, data_path, seq_len, use_mmap=False, mask_path=None, sequence_stride=1):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training array not found: '{data_path}'. Run preprocessing first.")
        if mask_path is not None and not os.path.exists(mask_path):
            raise FileNotFoundError(f"CRITICAL: SFT Mask Missing! Path exactly requested: '{mask_path}'. Silent fallback disabled to prevent irreversible model degradation.")
            
        if use_mmap:
            self.data = np.load(data_path, mmap_mode='r')
            self.mask = np.load(mask_path, mmap_mode='r') if mask_path else None
        else:
            self.data = np.load(data_path)
            self.mask = np.load(mask_path) if mask_path else None
            
        self.seq_len = seq_len
        self.is_sft = self.mask is not None
        self.sequence_stride = max(1, int(sequence_stride))
        self.indices = None
        window_count = max(0, len(self.data) - self.seq_len)
        candidate_starts = np.arange(0, window_count, self.sequence_stride, dtype=np.int64)

        if self.is_sft:
            if len(self.mask) != len(self.data):
                raise ValueError(f"SFT mask length ({len(self.mask)}) must match token array length ({len(self.data)}).")
            if window_count > 0:
                mask_arr = np.asarray(self.mask, dtype=np.int64)
                prefix = np.concatenate(([0], np.cumsum(mask_arr)))
                useful = prefix[candidate_starts + self.seq_len + 1] - prefix[candidate_starts + 1]
                self.indices = candidate_starts[useful > 0].astype(np.int64)
            else:
                self.indices = np.array([], dtype=np.int64)
        elif self.sequence_stride != 1:
            self.indices = candidate_starts

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        # Slice from the array (copy from mmap to avoid read-only tensor warnings)
        x = np.array(self.data[idx:idx + self.seq_len])
        y = np.array(self.data[idx + 1:idx + self.seq_len + 1])
        
        if self.is_sft:
            m = np.array(self.mask[idx + 1:idx + self.seq_len + 1])
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m).to(torch.long)
            
        return torch.from_numpy(x), torch.from_numpy(y)

def load_config(is_sft=False):
    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    
    # Unified Config Factory: Never let SFT parameters break Base architecture dimensions
    if is_sft and os.path.exists('configs/sft_config.ini'):
        sft_config = configparser.ConfigParser()
        sft_config.read('configs/sft_config.ini')
        for section in ['TRAINING', 'CHAT', 'DATA', 'MODEL', 'EVAL']:
            if sft_config.has_section(section):
                if not config.has_section(section):
                    config.add_section(section)
                for key, val in sft_config.items(section):
                    if section == 'MODEL' and key in STRUCTURAL_MODEL_KEYS:
                        continue # Block structural drift completely
                    config.set(section, key, val)
    return config

def validate_config(cfg, is_sft=False):
    for section in ['MODEL', 'TRAINING', 'DATA']:
        if not cfg.has_section(section):
            raise ValueError(f"Config missing required [{section}] section.")

    d_model = int(cfg['MODEL']['d_model'])
    n_heads = int(cfg['MODEL']['n_heads'])
    head_dim = d_model // n_heads
    vocab_size = int(cfg['MODEL']['vocab_size'])
    seq_len = int(cfg['MODEL']['max_seq_len'])
    hidden_mult = float(cfg['MODEL'].get('hidden_mult', 4.0))
    norm_eps = float(cfg['MODEL'].get('norm_eps', 1e-6))
    use_moe = cfg['MODEL'].getboolean('use_moe', fallback=False)
    num_experts = int(cfg['MODEL'].get('num_experts', 4))
    experts_per_tok = int(cfg['MODEL'].get('experts_per_token', 2))
    batch_size = int(cfg['TRAINING']['batch_size'])
    accum = int(cfg['TRAINING'].get('gradient_accumulation_steps', 1))
    epochs = int(cfg['TRAINING'].get('epochs', 1))
    lr = float(cfg['TRAINING']['lr'])
    min_lr = float(cfg['TRAINING'].get('min_lr', lr * 0.1))

    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim. Current head_dim is {head_dim}.")
    if seq_len < 2:
        raise ValueError("max_seq_len must be at least 2.")
    if hidden_mult <= 0:
        raise ValueError("MODEL hidden_mult must be > 0.")
    if norm_eps <= 0:
        raise ValueError("MODEL norm_eps must be > 0.")
    if use_moe:
        if num_experts < 1:
            raise ValueError("MODEL num_experts must be >= 1 when use_moe = True.")
        if experts_per_tok < 1 or experts_per_tok > num_experts:
            raise ValueError("MODEL experts_per_token must be between 1 and num_experts.")
    if batch_size < 1 or accum < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must both be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if lr <= 0 or min_lr < 0 or min_lr > lr:
        raise ValueError("Learning rates must satisfy lr > 0 and 0 <= min_lr <= lr.")

    sequence_stride = int(cfg['TRAINING'].get('sequence_stride', 0))
    val_sequence_stride = int(cfg['TRAINING'].get('val_sequence_stride', 0))
    if sequence_stride < 0 or val_sequence_stride < 0:
        raise ValueError("sequence_stride and val_sequence_stride must be >= 0. Use 0 for max_seq_len stride.")

    tokenizer = CharTokenizer()
    vocab_path = cfg['DATA'].get('vocab_path', '')
    if not tokenizer.load(vocab_path):
        raise FileNotFoundError(f"Tokenizer not found at '{vocab_path}'. Train the shared tokenizer first.")
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match config vocab_size ({vocab_size}).")

    for key in ['train_path', 'val_path']:
        path = cfg['DATA'].get(key, '')
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} not found at '{path}'. Run preprocessing first.")

    if is_sft:
        for key in ['train_path', 'val_path']:
            mask_path = cfg['DATA'][key].replace('.npy', '_mask.npy')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"SFT loss mask missing at '{mask_path}'. Run SFT preprocessing first.")

def build_adamw(param_groups, lr, device):
    if str(device).startswith('cuda'):
        try:
            return torch.optim.AdamW(param_groups, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(param_groups, lr=lr)

def compute_lm_loss(logits, targets, loss_mask, vocab_size):
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    if loss_mask is None:
        return F.cross_entropy(flat_logits, flat_targets)

    flat_mask = loss_mask.reshape(-1).to(torch.bool)

    if bool(flat_mask.all()):
        return F.cross_entropy(flat_logits, flat_targets)
    if not bool(flat_mask.any()):
        return flat_logits.sum() * 0.0

    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])

def move_batch_to_device(batch, device):
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device), None
    x, y, loss_mask = batch
    return x.to(device), y.to(device), loss_mask.to(device)

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

def maybe_compile_model(model, cfg, device):
    enabled = cfg.getboolean('TRAINING', 'enable_torch_compile', fallback=False)
    if not enabled:
        return model, False
    if not hasattr(torch, 'compile'):
        print("  [System] torch.compile is not available in this PyTorch build. Continuing without it.")
        return model, False

    backend = cfg['TRAINING'].get('torch_compile_backend', '').strip() or None
    mode = cfg['TRAINING'].get('torch_compile_mode', '').strip() or None
    try:
        compile_kwargs = {}
        if backend:
            compile_kwargs['backend'] = backend
        if mode:
            compile_kwargs['mode'] = mode
        compiled = torch.compile(model, **compile_kwargs)
        print(f"  [System] torch.compile enabled ({backend or 'default'}, {mode or 'default'}).")
        return compiled, True
    except Exception as e:
        print(f"  [System] torch.compile disabled after startup failure: {e}")
        return model, False

def append_run_log(log_path, payload):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

def generate_eval_sample(model, tokenizer, prompt, device, max_seq_len, max_tokens=40):
    model.eval()
    raw_prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = [tokenizer.bos_id] + raw_prompt_tokens
    if len(prompt_tokens) >= max_seq_len:
        prompt_tokens = [tokenizer.bos_id] + raw_prompt_tokens[-(max_seq_len - 2):]

    generated = []
    model.reset_cache()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits, _ = model(input_ids, use_cache=True)
        next_token = int(torch.argmax(logits[0, -1]).item())
        for _ in range(max_tokens):
            if next_token == tokenizer.eos_id or len(prompt_tokens) + len(generated) >= max_seq_len:
                break
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, _ = model(input_ids, use_cache=True)
            next_token = int(torch.argmax(logits[0, -1]).item())

    return tokenizer.decode(generated).strip()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_param_count(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

def train_model(resume_from=None, is_sft=False):
    cfg = load_config(is_sft)
    validate_config(cfg, is_sft=is_sft)

    device = "cpu"

    # CPU Thread Optimization: lock to physical cores to prevent hyper-threading cache thrashing
    num_threads = int(cfg['TRAINING'].get('num_threads', 0))
    safe_set_torch_threads(num_threads)

    batch_size = int(cfg['TRAINING']['batch_size'])
    seq_len = int(cfg['MODEL']['max_seq_len'])
    epochs = int(cfg['TRAINING']['epochs'])
    lr = float(cfg['TRAINING']['lr'])
    weight_decay = float(cfg['TRAINING'].get('weight_decay', 0.01))

    gradient_accumulation_steps = int(cfg['TRAINING'].get('gradient_accumulation_steps', 1))
    early_stopping_patience = int(cfg['TRAINING'].get('early_stopping_patience', 5))
    moe_aux_loss_coef = float(cfg['TRAINING'].get('moe_aux_loss_coef', 0.01))
    
    use_mmap = cfg['TRAINING'].getboolean('use_mmap', fallback=False)
    num_workers = int(cfg['TRAINING'].get('num_workers', 0))
    sequence_stride = int(cfg['TRAINING'].get('sequence_stride', 0))
    if sequence_stride <= 0:
        sequence_stride = seq_len
    val_sequence_stride = int(cfg['TRAINING'].get('val_sequence_stride', 0))
    if val_sequence_stride <= 0:
        val_sequence_stride = sequence_stride

    print("Loading datasets...")
    if is_sft:
        train_mask_path = cfg['DATA']['train_path'].replace('.npy', '_mask.npy')
        val_mask_path = cfg['DATA']['val_path'].replace('.npy', '_mask.npy')
    else:
        train_mask_path = None
        val_mask_path = None
        
    train_dataset = TokenDataset(cfg['DATA']['train_path'], seq_len, use_mmap=use_mmap, mask_path=train_mask_path, sequence_stride=sequence_stride)
    val_dataset = TokenDataset(cfg['DATA']['val_path'], seq_len, use_mmap=use_mmap, mask_path=val_mask_path, sequence_stride=val_sequence_stride)

    if len(train_dataset) < batch_size and sequence_stride != 1:
        print("Packed windows are fewer than one batch. Falling back to dense stride=1 for this run.")
        sequence_stride = 1
        train_dataset = TokenDataset(cfg['DATA']['train_path'], seq_len, use_mmap=use_mmap, mask_path=train_mask_path, sequence_stride=sequence_stride)
    
    loader_kwargs = {'num_workers': num_workers}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg['TRAINING'].getboolean('persistent_workers', fallback=True)
        loader_kwargs['prefetch_factor'] = int(cfg['TRAINING'].get('prefetch_factor', 2))

    # Accelerated Loading
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        **loader_kwargs
    )

    if len(train_loader) == 0:
        print("Error: Training set is too small to form even one batch. Add more data or reduce batch_size.")
        return

    vocab_size = int(cfg['MODEL']['vocab_size'])
    model = Transformer(cfg['MODEL']).to(device)
    eval_tokenizer = None
    eval_prompt = cfg.get('EVAL', 'sample_prompt', fallback='<human>: Hello\\n<gpt>:').replace('\\n', '\n')
    eval_enabled = cfg.getboolean('EVAL', 'enable_sample_eval', fallback=False)
    eval_max_tokens = int(cfg.get('EVAL', 'sample_max_tokens', fallback='40'))
    if eval_enabled:
        eval_tokenizer = CharTokenizer()
        eval_tokenizer.load(cfg['DATA']['vocab_path'])

    # Shield 1D parameters (RMSNorm, biases) and embeddings from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'emb' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = build_adamw(optim_groups, lr=lr, device=device)

    # Scheduler tracks actual optimizer steps rigorously mapped to accumulation thresholds
    steps_per_epoch = max(1, math.ceil(len(train_loader) / gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(cfg['TRAINING'].get('warmup_ratio', 0.05))
    warmup_steps = min(max(10, int(warmup_ratio * total_steps)), total_steps, 1000)
    
    min_lr = float(cfg['TRAINING'].get('min_lr', lr * 0.1))
    lr_decay_ratio = min_lr / lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
            
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        decay_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_decay_ratio + decay_factor * (1.0 - lr_decay_ratio)

    checkpoint_dir = os.path.dirname(cfg['TRAINING']['checkpoint_path'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = cfg['TRAINING']['checkpoint_path']
    run_log_path = os.path.join(checkpoint_dir, "training_log.jsonl")
    
    init_source = "scratch"
    resume_optimizer = False
    if is_sft:
        base_cfg = configparser.ConfigParser()
        base_cfg.read('configs/config.ini')
        base_model_path = base_cfg['TRAINING'].get('checkpoint_path', 'model/best_model.pth')
        resume_sft_checkpoint = cfg['TRAINING'].getboolean('resume_sft_checkpoint', fallback=False)
    else:
        base_model_path = None
        resume_sft_checkpoint = False
    
    if resume_from:
        resume_path = resume_from
        init_source = f"explicit resume: {resume_path}"
        resume_optimizer = True
    elif is_sft:
        if resume_sft_checkpoint and os.path.exists(best_model_path):
            resume_path = best_model_path
            init_source = f"SFT checkpoint: {best_model_path}"
            resume_optimizer = True
        elif os.path.exists(base_model_path):
            resume_path = base_model_path
            init_source = f"base checkpoint: {base_model_path}"
            resume_optimizer = False
            if os.path.exists(best_model_path):
                print(f"  [System] Existing SFT checkpoint ignored because resume_sft_checkpoint = False: {best_model_path}")
            print(f"  [System] Initiating SFT from Base Weights: {base_model_path}")
        else:
            resume_path = None
            print("  WARNING: Base weights not found! SFT will train from scratch (Not Recommended!)")
    elif os.path.exists(best_model_path):
        resume_path = best_model_path
        init_source = f"checkpoint: {best_model_path}"
        resume_optimizer = True
    else:
        resume_path = None

    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0

    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            
            # Base-to-SFT bridge loads weights only; SFT optimizer/schedule starts fresh.
            if resume_optimizer:
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
            else:
                print("  [System] Base Model weights loaded. SFT Optimizer will build a fresh state.")
        except Exception as e:
            print(f"Warning: Could not resume from {resume_path}: {e}")
            print("Starting fresh.")
            
    # CRITICAL: Scheduler initialized specifically with last_epoch mapping to prevent Warmup replay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=(start_epoch * steps_per_epoch) - 1 if start_epoch > 0 else -1)
    
    if resume_path and os.path.exists(resume_path) and resume_optimizer:
        if 'checkpoint' in locals() and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    run_model, compile_active = maybe_compile_model(model, cfg, device)

    num_params = sum(p.numel() for p in model.parameters())
    d_model = cfg['MODEL'].get('d_model', '?')
    n_layers = cfg['MODEL'].get('n_layers', '?')
    n_heads = cfg['MODEL'].get('n_heads', '?')

    mode_tag = "SFT" if is_sft else "Pretrain"
    overfit_monitor = OverfitMonitor(mode_tag)
    
    try:
        import psutil
    except ImportError:
        psutil = None
    initial_ram = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0.0

    clear_screen()
    print_run_banner(f"{mode_tag} Training Run", [
        ("🧠 Model", f"{format_param_count(num_params)} params  │  {n_layers}L {n_heads}H d={d_model}"),
        ("⚙ Config", f"{device.upper()}  │  Batch: {batch_size}×{gradient_accumulation_steps} (eff {batch_size*gradient_accumulation_steps})  │  LR: {lr}"),
        ("📊 Data", f"Train: {len(train_dataset):,} windows  │  Val: {len(val_dataset):,} windows"),
        ("📐 Schedule", f"{total_steps:,} steps  │  Warmup: {warmup_steps}  │  Min LR: {min_lr}"),
        ("⚡ Perf", f"Stride: {sequence_stride}/{val_sequence_stride}  │  Checkpoint: {cfg['MODEL'].get('gradient_checkpointing', 'False')}  │  Compile: {compile_active}"),
        ("💾 Memory", f"RAM: {initial_ram:.0f} MB"),
    ])

    if best_val_loss != float('inf'):
        best_ppl = math.exp(min(best_val_loss, 20.0))
        print(f"  📈 Resuming    Epoch {start_epoch+1}  │  Best Loss: {best_val_loss:.4f}  │  PPL: {best_ppl:.2f}")
    elif is_sft and resume_path and os.path.exists(resume_path):
        print(f"  📈 Status      SFT initialized from {init_source}")
    else:
        print(f"  📈 Status      {mode_tag} from scratch")
    print("─" * 48)

    append_run_log(run_log_path, {
        'event': 'run_start',
        'mode': mode_tag,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'params': int(num_params),
        'config': {
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'lr': lr,
            'min_lr': min_lr,
            'sequence_stride': sequence_stride,
            'val_sequence_stride': val_sequence_stride,
            'gradient_checkpointing': cfg['MODEL'].get('gradient_checkpointing', 'False'),
            'torch_compile': compile_active,
            'init_source': init_source,
            'resume_optimizer': resume_optimizer,
        }
    })

    try:
        for epoch in range(start_epoch, epochs):
            run_model.train()
            total_loss = 0
            ema_loss = 0.0
            num_steps = 0

            optimizer.zero_grad(set_to_none=True)

            bar_fmt = "  Epoch {desc} │{bar}│ {percentage:3.0f}%  {postfix}"
            pbar = tqdm(train_loader, desc=f"{epoch+1}/{epochs}", leave=False, bar_format=bar_fmt, ncols=90)
            pbar.set_postfix_str("Loss: -.---- │ PPL: ------ │ ---- Tok/s")

            step_t0 = time.perf_counter()
            ema_tok_s = 0.0
            last_mem_time = 0.0
            ram_mb = 0.0

            for step, batch in enumerate(pbar):
                # Update System Memory & Window Title every 1.0 seconds
                if time.perf_counter() - last_mem_time > 1.0:
                    ram_mb = psutil.Process().memory_info().rss / 1024 / 1024 if psutil else 0.0
                    last_mem_time = time.perf_counter()
                    
                    title = f"Axiom AI Training  |  RAM: {ram_mb:.0f} MB"
                    if os.name == 'nt':
                        import ctypes
                        ctypes.windll.kernel32.SetConsoleTitleW(title)
                    else:
                        sys.stdout.write(f"\033]0;{title}\007")
                        sys.stdout.flush()

                x, y, loss_mask = move_batch_to_device(batch, device)

                logits, aux_loss = run_model(x)
                loss = compute_lm_loss(logits, y, loss_mask, vocab_size)
                
                if isinstance(aux_loss, torch.Tensor) or aux_loss != 0.0:
                    loss = loss + (moe_aux_loss_coef * aux_loss)
                
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()

                step_loss = loss.item()
                total_loss += step_loss
                num_steps += 1

                # EMA for responsive real-time display
                if num_steps == 1:
                    ema_loss = step_loss
                else:
                    ema_loss = 0.95 * ema_loss + 0.05 * step_loss

                # Speed Profiling (Tokens / Second)
                step_t1 = time.perf_counter()
                dt = step_t1 - step_t0
                step_t0 = step_t1
                tokens_processed = x.numel()  # Batch Size * Seq Len
                tok_per_sec = tokens_processed / max(dt, 1e-6)

                if num_steps == 1:
                    ema_tok_s = tok_per_sec
                else:
                    ema_tok_s = 0.95 * ema_tok_s + 0.05 * tok_per_sec

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if (step + 1) % 5 == 0 or step == 0:
                    cur_ppl = math.exp(min(ema_loss, 20.0))
                    pbar.set_postfix_str(f"Loss: {ema_loss:.4f} │ PPL: {cur_ppl:>6.1f} │ Tok/s: {ema_tok_s:>4.0f} │ RAM: {ram_mb:>4.0f}M")

            pbar.close()

            avg_train_loss = total_loss / max(1, num_steps)
            train_ppl = math.exp(min(avg_train_loss, 20.0))

            # --- Validation ---
            run_model.eval()
            val_loss_sum = 0
            val_batches = 0
            max_val_steps = int(cfg['TRAINING'].get('max_val_steps', 200))  # Dynamically bound from config
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y, loss_mask = move_batch_to_device(batch, device)
                    logits, _ = run_model(x)
                    loss = compute_lm_loss(logits, y, loss_mask, vocab_size)
                    
                    val_loss_sum += loss.item()
                    val_batches += 1
                    
                    if val_batches >= max_val_steps:
                        break

            if val_batches == 0:
                avg_val_loss = float('inf')
                val_ppl = float('inf')
                val_str = "V Loss: N/A (val set too small)"
            else:
                avg_val_loss = val_loss_sum / val_batches
                val_ppl = math.exp(min(avg_val_loss, 20.0))
                val_str = f"V Loss {avg_val_loss:.4f} (PPL {val_ppl:.1f})"

            # Epoch summary line
            train_str = f"T Loss {avg_train_loss:.4f} (PPL {train_ppl:.1f})"
            tag = ""

            checkpoint_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': float(min(best_val_loss, avg_val_loss)),
                'config': dict(cfg['MODEL']),
                'mode': mode_tag,
                'init_source': init_source,
                'resume_optimizer': resume_optimizer,
            }

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save(checkpoint_data, best_model_path)
                tag = "  ✨ New Best"
            else:
                early_stopping_counter += 1

            print(f"  Epoch {epoch+1:<3} │ {train_str}  │  {val_str}{tag}")
            overfit_warning = overfit_monitor.update(epoch + 1, avg_train_loss, avg_val_loss)
            if overfit_warning:
                print(overfit_warning)
            append_run_log(run_log_path, {
                'event': 'epoch',
                'mode': mode_tag,
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'train_ppl': float(train_ppl),
                'val_loss': float(avg_val_loss) if avg_val_loss != float('inf') else None,
                'val_ppl': float(val_ppl) if val_batches > 0 else None,
                'lr': float(scheduler.get_last_lr()[0]),
                'best': bool(tag),
            })

            if eval_enabled and eval_tokenizer is not None:
                sample = generate_eval_sample(model, eval_tokenizer, eval_prompt, device, seq_len, max_tokens=eval_max_tokens)
                print(f"  Eval Sample │ {sample[:180]}")
                append_run_log(run_log_path, {
                    'event': 'sample',
                    'mode': mode_tag,
                    'epoch': epoch + 1,
                    'prompt': eval_prompt,
                    'sample': sample,
                })
                run_model.train()

            if early_stopping_counter >= early_stopping_patience:
                print(f"\n  ⛔ Early stopping after {early_stopping_counter} epochs without improvement.")
                break

            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        print("─" * 48)
        if best_val_loss != float('inf'):
            final_ppl = math.exp(min(best_val_loss, 20.0))
            print(f"  ✅ Done │ Best Val Loss: {best_val_loss:.4f}  │  PPL: {final_ppl:.2f}")
        else:
            print("  ✅ Done │ No validation improvement recorded.")
            
    except KeyboardInterrupt:
        # Prevent pbar format destruction
        try: pbar.close() 
        except: pass
        
        print("\n\n  ⚠️  [System] Ctrl+C User Interrupt Detected!")
        print(f"  💾 Gracefully halting and sweeping state dict to {best_model_path} ...")
        
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # Safely capture epoch if it exists, otherwise fall back to start
            'epoch': epoch if 'epoch' in locals() else start_epoch,
            'best_val_loss': float(best_val_loss),
            'config': dict(cfg['MODEL']),
            'mode': mode_tag,
            'init_source': init_source,
            'resume_optimizer': resume_optimizer,
        }
        torch.save(checkpoint_data, best_model_path)
        print("  ✅ Emergency Checkpoint secured. Returning to menu.\n")
        time.sleep(1.5)
        return

if __name__ == "__main__":
    train_model()
