import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        # Stage 2: The Ears
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, input_ids):
        B, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        
        # Determine padding map (0 marks the pad array sequence elements)
        pad_mask = (input_ids == 0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return x
