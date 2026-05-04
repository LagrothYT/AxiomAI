import torch
import torch.nn as nn
from model import TransformerBlock

class TextEncoder(nn.Module):
    """
    A basic trainable Transformer-based text encoder.
    Uses the existing AxiomAI TransformerBlock architecture to ensure consistency.
    """
    def __init__(self, config):
        super().__init__()
        # config is from [TEXT_ENCODER] section of video_config.ini
        d_model = int(config.get('d_model', 128))
        n_layers = int(config.get('n_layers', 2))
        n_heads = int(config.get('n_heads', 4))
        vocab_size = int(config.get('vocab_size', 2096))
        max_seq_len = int(config.get('max_seq_len', 64))
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_causal=False):
        # x is (B, SeqLen) tokens
        b, s = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :s, :]
        
        for block in self.blocks:
            # TransformerBlock in model.py returns (x, aux_loss)
            h, _ = block(h, is_causal=is_causal)
            
        return self.norm(h) # (B, SeqLen, D)
