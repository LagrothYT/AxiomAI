import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm, rotate_half

def apply_3d_rope(q, k, t_coords, h_coords, w_coords, cos_buf, sin_buf):
    """
    Applies 3D Rotary Positional Encoding by partitioning the head dimension.
    """
    b, n_heads, s, head_dim = q.shape
    d_t = head_dim // 4
    d_h = (head_dim - d_t) // 2
    d_w = head_dim - d_t - d_h
    
    def apply_seg(x, coords, d_start, d_end):
        seg = x[..., d_start:d_end]
        width = d_end - d_start
        c = cos_buf[coords][..., :width].unsqueeze(1)
        s_buf = sin_buf[coords][..., :width].unsqueeze(1)
        return (seg * c) + (rotate_half(seg) * s_buf)

    q_t = apply_seg(q, t_coords, 0, d_t)
    q_h = apply_seg(q, h_coords, d_t, d_t + d_h)
    q_w = apply_seg(q, w_coords, d_t + d_h, head_dim)
    
    k_t = apply_seg(k, t_coords, 0, d_t)
    k_h = apply_seg(k, h_coords, d_t, d_t + d_h)
    k_w = apply_seg(k, w_coords, d_t + d_h, head_dim)
    
    return torch.cat([q_t, q_h, q_w], dim=-1), torch.cat([k_t, k_h, k_w], dim=-1)

class FactorizedAttention(nn.Module):
    """
    Factorized Attention executing dynamically reshaped spatial and temporal passes.
    """
    def __init__(self, d_model, n_heads, mode):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"Video attention d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.mode = mode
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, seq_lens, t_coords, h_coords, w_coords, cos_buf, sin_buf, use_cache=False, kv_cache=None):
        b, s, d = x.shape
        t_len, h_len, w_len = seq_lens
        
        # Linear projection
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply 3D RoPE strictly to the NEW logical projections
        q, k = apply_3d_rope(q, k, t_coords, h_coords, w_coords, cos_buf, sin_buf)

        # Spatial attention is intra-frame only, so retaining spatial KV cache just wastes RAM.
        if self.mode != 'spatial' and use_cache and kv_cache is not None:
            # Aggregate the sequence exclusively after injecting structural coordinates
            # This deletes the requirement to dynamically zero-pad historical sequences
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        
        new_kv_cache = (k, v) if (use_cache and self.mode != 'spatial') else None

        if self.mode == 'spatial':
            # Spatial Attention is STRICTLY INTRA-FRAME. Past frames are entirely irrelevant.
            # Thus, we only use the current query sequence block (S) and ignore the past KV Cache!
            curr_t = q.shape[2] // (h_len * w_len)
            
            # Reshape to (B*curr_t, Heads, H*W, D)
            q_s = q.view(b, self.n_heads, curr_t, h_len*w_len, self.head_dim).transpose(1, 2).reshape(b*curr_t, self.n_heads, h_len*w_len, self.head_dim)
            k_s = k[..., -curr_t*(h_len*w_len):].view(b, self.n_heads, curr_t, h_len*w_len, self.head_dim).transpose(1, 2).reshape(b*curr_t, self.n_heads, h_len*w_len, self.head_dim)
            v_s = v[..., -curr_t*(h_len*w_len):].view(b, self.n_heads, curr_t, h_len*w_len, self.head_dim).transpose(1, 2).reshape(b*curr_t, self.n_heads, h_len*w_len, self.head_dim)
            
            # Spatial Attention is ALWAYS bidirectional (is_causal=False) inside the frame!
            out = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False)
            out = out.view(b, curr_t, self.n_heads, h_len*w_len, self.head_dim).transpose(1, 2).reshape(b, self.n_heads, s, self.head_dim)
                
        elif self.mode == 'temporal':
            # Temporal Attention spans ACROSS frames. 
            curr_t_q = q.shape[2] // (h_len * w_len)
            curr_t_k = k.shape[2] // (h_len * w_len)
            
            # Reshape S to T, H*W -> (B*H*W, Heads, T, D)
            q_t = q.view(b, self.n_heads, curr_t_q, h_len*w_len, self.head_dim).permute(0, 3, 1, 2, 4).reshape(b*h_len*w_len, self.n_heads, curr_t_q, self.head_dim)
            k_t = k.view(b, self.n_heads, curr_t_k, h_len*w_len, self.head_dim).permute(0, 3, 1, 2, 4).reshape(b*h_len*w_len, self.n_heads, curr_t_k, self.head_dim)
            v_t = v.view(b, self.n_heads, curr_t_k, h_len*w_len, self.head_dim).permute(0, 3, 1, 2, 4).reshape(b*h_len*w_len, self.n_heads, curr_t_k, self.head_dim)
            
            # Full training is causal. Incremental generation already passes only past+current KV,
            # so a causal mask would incorrectly hide usable history from the single current query.
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=not use_cache)
            out = out.view(b, h_len*w_len, self.n_heads, curr_t_q, self.head_dim).permute(0, 2, 3, 1, 4).reshape(b, self.n_heads, s, self.head_dim)
                
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv_cache

class CrossAttention(nn.Module):
    """
    Standard Cross-Attention to inject text encoder embeddings.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False) # Context
        self.wv = nn.Linear(d_model, d_model, bias=False) # Context
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, context):
        b, s, d = x.shape
        ctx_b, ctx_s, _ = context.shape
        
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(context).view(ctx_b, ctx_s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(context).view(ctx_b, ctx_s, self.n_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out)

class STTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, config):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.spatial_attn = FactorizedAttention(d_model, n_heads, mode='spatial')
        
        self.norm2 = RMSNorm(d_model)
        self.temporal_attn = FactorizedAttention(d_model, n_heads, mode='temporal')
        
        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        
        self.norm3 = RMSNorm(d_model)
        hidden_dim = int(d_model * float(config.get('hidden_mult', 4.0)))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False)
        )

    def forward(self, x, context, seq_lens, t_coords, h_coords, w_coords, cos_buf, sin_buf, use_cache=False, kv_cache=None):
        if kv_cache is None: kv_cache = (None, None)
        
        s_out, s_cache = self.spatial_attn(self.norm1(x), seq_lens, t_coords, h_coords, w_coords, cos_buf, sin_buf, use_cache, kv_cache[0])
        x = x + s_out
        
        t_out, t_cache = self.temporal_attn(self.norm2(x), seq_lens, t_coords, h_coords, w_coords, cos_buf, sin_buf, use_cache, kv_cache[1])
        x = x + t_out
        
        if context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)
            
        x = x + self.ffn(self.norm3(x))
        return x, (s_cache, t_cache)

class MultiScaleVideoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = int(config.get('d_model', 128))
        n_layers = int(config.get('n_layers', 4))
        n_heads = int(config.get('n_heads', 8))
        codebook_size = int(config.get('codebook_size', 4096))
        if d_model % n_heads != 0:
            raise ValueError(f"VIDEO_MODEL d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        
        # Explicit BOS isolation logic structurally segregating Generation Token bounds from Spatial Quantization Bounds
        self.bos_id = int(config.get('bos_id', codebook_size))
        if self.bos_id < codebook_size:
            raise ValueError(f"VIDEO_MODEL bos_id ({self.bos_id}) must be >= VAE codebook_size ({codebook_size}).")
        max_seq = int(config.get('max_seq_len', 2048))
        max_scale_levels = int(config.get('max_scale_levels', 4))
        
        # Predict discrete tokens! Frame token logic cleanly decoupled from dictionary constraints
        self.tok_embed = nn.Embedding(self.bos_id + 1, d_model)
        self.scale_embed = nn.Embedding(max_scale_levels, d_model)
        
        self.blocks = nn.ModuleList([
            STTransformerBlock(d_model, n_heads, config) for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        # Output exactly codebook size logits
        self.head = nn.Linear(d_model, codebook_size, bias=False)

        head_dim = d_model // n_heads
        d_t = head_dim // 4
        d_h = (head_dim - d_t) // 2
        d_w = head_dim - d_t - d_h
        if min(d_t, d_h, d_w) < 2 or d_t % 2 or d_h % 2 or d_w % 2:
            raise ValueError(f"3D RoPE needs even non-empty temporal/spatial head splits. head_dim={head_dim} gives {(d_t, d_h, d_w)}.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('rope_cos', emb.cos(), persistent=False)
        self.register_buffer('rope_sin', emb.sin(), persistent=False)

    def forward(self, indices, context, scale_id, seq_lens, t_coords, h_coords, w_coords, use_cache=False, kv_caches=None):
        # Allow passing full seq directly (useful for caching during generation)
        x = self.tok_embed(indices)
        
        # Scale embedding
        x = x + self.scale_embed(scale_id).unsqueeze(1)
        
        new_caches = []
        if kv_caches is None: kv_caches = [(None, None)] * len(self.blocks)
        
        for block, cache in zip(self.blocks, kv_caches):
            x, new_cache = block(x, context, seq_lens, t_coords, h_coords, w_coords, self.rope_cos, self.rope_sin, use_cache, cache)
            new_caches.append(new_cache)
            
        return self.head(self.norm_f(x)), new_caches
