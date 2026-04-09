import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32 if in_channels >= 32 else in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32 if out_channels >= 32 else out_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.residual(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, hidden_dims=[64, 128, 256]):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1))
        
        current_dims = hidden_dims[0]
        for idx, dim in enumerate(hidden_dims):
            modules.append(ResBlock(current_dims, dim))
            current_dims = dim
            if idx < len(hidden_dims) - 1:
                modules.append(Downsample(current_dims))
                
        modules.append(ResBlock(current_dims, current_dims))
        modules.append(nn.GroupNorm(32, current_dims))
        modules.append(nn.SiLU())
        modules.append(nn.Conv2d(current_dims, latent_channels * 2, kernel_size=3, padding=1))
        
        self.encoder = nn.Sequential(*modules)
        
        current_dims = hidden_dims[-1]
        modules_dec = []
        modules_dec.append(nn.Conv2d(latent_channels, current_dims, kernel_size=3, padding=1))
        modules_dec.append(ResBlock(current_dims, current_dims))
        
        rev_hidden_dims = list(reversed(hidden_dims))
        for idx, dim in enumerate(rev_hidden_dims):
            modules_dec.append(ResBlock(current_dims, dim))
            current_dims = dim
            if idx < len(rev_hidden_dims) - 1:
                modules_dec.append(Upsample(current_dims))
                
        modules_dec.append(nn.GroupNorm(32, current_dims))
        modules_dec.append(nn.SiLU())
        modules_dec.append(nn.Conv2d(current_dims, in_channels, kernel_size=3, padding=1))
        
        self.decoder = nn.Sequential(*modules_dec)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
        
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
