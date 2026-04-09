import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x, context):
        b, c, h, w = x.shape
        # Flatten spatial dimensions: (b, c, h, w) -> (b, h*w, c)
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.reshape(b, -1, self.heads, q.shape[-1] // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, self.heads, k.shape[-1] // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, self.heads, v.shape[-1] // self.heads).permute(0, 2, 1, 3)
        
        # Attention mathematically fuses visual spatial elements mapped against grammar sequences
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, self.heads * q.shape[-1])
        out = self.to_out(out)
        
        # Reshape back to spatial tensor map
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return out + x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        groups = 8 if out_channels >= 8 else out_channels
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        
        self.attn = CrossAttention(out_channels, context_dim)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb, context):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        time_e = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_e
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Spatial tensor mapping 'listens' to text context
        h = self.attn(h, context)
        
        return h + self.res_conv(x)

class LatentDiffusion(nn.Module):
    def __init__(self, in_channels=4, text_embed_dim=256, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.down1 = UNetBlock(in_channels, 64, time_embed_dim, text_embed_dim)
        self.down2 = UNetBlock(64, 128, time_embed_dim, text_embed_dim)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.mid = UNetBlock(128, 128, time_embed_dim, text_embed_dim)
        
        self.up2 = UNetBlock(128 + 64, 64, time_embed_dim, text_embed_dim)
        self.up1 = UNetBlock(64, in_channels, time_embed_dim, text_embed_dim)
        
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t, context):
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        
        d1 = self.down1(x, t_emb, context)
        p1 = self.pool(d1)
        d2 = self.down2(p1, t_emb, context)
        
        m = self.mid(d2, t_emb, context)
        
        u2 = self.upsample(m)
        u2 = torch.cat([u2, d1], dim=1) # Explicit spatial skip mapping
        u2 = self.up2(u2, t_emb, context)
        
        out = self.up1(u2, t_emb, context)
        return self.final_conv(out)
