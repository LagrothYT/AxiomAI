import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
except ImportError:
    torch_checkpoint = None

def cfg_get_bool(cfg, key, default=False):
    if hasattr(cfg, 'getboolean'):
        return cfg.getboolean(key, fallback=default)
    val = cfg.get(key, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')

def checkpoint_block(fn, x):
    if torch_checkpoint is None:
        return fn(x)
    try:
        return torch_checkpoint(fn, x, use_reentrant=False)
    except TypeError:
        return torch_checkpoint(fn, x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm) * self.weight

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos_buf, sin_buf, offset=0):
    """Apply precomputed RoPE by slicing into cached cos/sin buffers."""
    seq_len = q.shape[2]
    cos = cos_buf[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
    sin = sin_buf[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=256):
        super().__init__()
        
        # Enforce inviolable RoPE conditions to prevent silent memory/tensor layout corruption
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be perfectly divisible by n_heads ({n_heads})"
        assert (d_model // n_heads) % 2 == 0, f"head_dim ({d_model // n_heads}) must be strictly even to cleanly split for RoPE rotation"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.cache_k = None
        self.cache_v = None

        # Precompute RoPE sin/cos buffers once
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer('rope_cos', emb.cos(), persistent=False)
        self.register_buffer('rope_sin', emb.sin(), persistent=False)

    def forward(self, x, use_cache=False, is_causal=True):
        b, s, d = x.shape
        
        # Calculate correct RoPE offset for cached past tokens
        past_len = 0
        if use_cache and self.cache_k is not None:
            past_len = self.cache_k.shape[2]
            
        if past_len + s > self.rope_cos.size(0):
            raise RuntimeError(f"Forward pass sequence length ({past_len + s}) critically exceeds initialized RoPE buffer ({self.rope_cos.size(0)}).")
     
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin, offset=past_len)

        # Concatenate cache (already RoPE'd) + new tokens
        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim=2)
            v = torch.cat([self.cache_v, v], dim=2)

        kv_len = k.shape[2]
        
        # PyTorch 2.0+ Accelerated Fused Attention Kernel
        # Instantly bypasses Python overhead and eliminates massive 'mask' allocations.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and (s > 1))

        out = out.transpose(1, 2).contiguous().view(b, s, d)

        if use_cache:
            self.cache_k = k
            self.cache_v = v

        return self.wo(out)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_mult=4):
        super().__init__()
        hidden_dim = int(d_model * hidden_mult)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, d_model, hidden_mult, num_experts, experts_per_tok):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_tok = experts_per_tok
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model, hidden_mult) for _ in range(num_experts)])

    def forward(self, x):
        batch, seq, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Calculate Auxiliary Load Balancing Loss
        P_bar = routing_probs.mean(dim=0)
        routing_weights, selected_experts = torch.topk(routing_probs, self.experts_per_tok, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Calculating Load Balancing algebraically prevents allocating 
        # multi-dimensional dense torch.zeros() footprint to system RAM
        expert_counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        f_bar = expert_counts.float() / (batch * seq * self.experts_per_tok)
        aux_loss = self.num_experts * torch.sum(P_bar * f_bar)
        
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, kth_expert = torch.where(selected_experts == i)
            if token_indices.numel() > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert(expert_inputs)
                weights_for_this_expert = routing_weights[token_indices, kth_expert].unsqueeze(-1)
                expert_outputs = expert_outputs * weights_for_this_expert
                out_flat.index_add_(0, token_indices, expert_outputs)
                
        return out_flat.view(batch, seq, d_model), aux_loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=256, hidden_mult=4.0, eps=1e-6, use_moe=False, num_experts=4, experts_per_tok=2):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.attn = Attention(d_model, n_heads, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoELayer(d_model, hidden_mult, num_experts, experts_per_tok)
        else:
            self.ffn = FeedForward(d_model, hidden_mult=hidden_mult)

    def forward(self, x, use_cache=False, is_causal=True):
        x = x + self.attn(self.norm1(x), use_cache=use_cache, is_causal=is_causal)
        aux_loss = x.new_zeros(())
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
        else:
            x = x + self.ffn(self.norm2(x))
        return x, aux_loss

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = int(cfg['d_model'])
        n_layers = int(cfg['n_layers'])
        n_heads = int(cfg['n_heads'])
        vocab_size = int(cfg['vocab_size'])
        max_seq_len = int(cfg.get('max_seq_len', 256))
        hidden_mult = float(cfg.get('hidden_mult', 4.0))
        norm_eps = float(cfg.get('norm_eps', 1e-6))
        self.gradient_checkpointing = cfg_get_bool(cfg, 'gradient_checkpointing', False)
        
        # MoE parameters
        use_moe = cfg_get_bool(cfg, 'use_moe', False)
        num_experts = int(cfg.get('num_experts', 4))
        experts_per_tok = int(cfg.get('experts_per_token', 2))
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len=max_seq_len, hidden_mult=hidden_mult, eps=norm_eps,
                             use_moe=use_moe, num_experts=num_experts, experts_per_tok=experts_per_tok) 
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model, eps=norm_eps)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Structure Hardening: Weight Tying
        # Forces embeddings to map bidirectionally and deletes massive parameter redundancy
        self.output.weight = self.tok_emb.weight
        
        # Run Gaussian initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, use_cache=False, is_causal=True):
        h = self.tok_emb(x)
        total_aux_loss = 0.0
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and not use_cache:
                def block_forward(hidden, block=block):
                    return block(hidden, use_cache=False, is_causal=is_causal)
                h, aux_loss = checkpoint_block(block_forward, h)
            else:
                h, aux_loss = block(h, use_cache=use_cache, is_causal=is_causal)
            total_aux_loss += aux_loss
            
        h = self.norm_f(h)
        return self.output(h), total_aux_loss

    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()
