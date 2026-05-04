import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    """
    3D Convolution that is 'causal' in the temporal dimension.
    Padding is only applied to the beginning of the time dimension to ensure
    future frames do not influence past frames.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__()
        # kernel_size is expected to be (kT, kH, kW)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
            
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Temporal padding (kT - 1) * dilation[0] at the start, 0 at the end
        # Spatial padding (kH // 2, kW // 2) on both sides
        self.pad = (kernel_size[2]//2, kernel_size[2]//2, 
                    kernel_size[1]//2, kernel_size[1]//2, 
                    kernel_size[0]-1, 0)
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, **kwargs)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = F.pad(x, self.pad)
        return self.conv(x)

class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, residual_scale=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_scale = float(residual_scale)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h * self.residual_scale

class VectorQuantizer(nn.Module):
    """
    Discretizes the continuous latent space into a finite vocabulary (codebook).
    Required for autoregressive next-token prediction.
    """
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        chunk_size=2048,
        use_ema=True,
        ema_decay=0.99,
        ema_eps=1e-5,
        dead_code_refresh=False,
        dead_code_threshold=0.01,
        max_code_refresh=64,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.chunk_size = int(chunk_size)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.dead_code_refresh = bool(dead_code_refresh)
        self.dead_code_threshold = float(dead_code_threshold)
        self.max_code_refresh = int(max_code_refresh)
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.embedding.weight.requires_grad_(not self.use_ema)
        self.register_buffer("ema_cluster_size", torch.ones(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def _nearest_embedding_indices(self, inputs_flat):
        # Squared L2 distance is enough for argmin and avoids torch.cdist sqrt overhead.
        emb = self.embedding.weight
        emb_norm = emb.pow(2).sum(dim=1).unsqueeze(0)

        def nearest(chunk):
            chunk_norm = chunk.pow(2).sum(dim=1, keepdim=True)
            distances = chunk_norm + emb_norm - 2.0 * chunk @ emb.t()
            return torch.argmin(distances, dim=1)

        if self.chunk_size <= 0 or inputs_flat.size(0) <= self.chunk_size:
            return nearest(inputs_flat)

        parts = []
        for start in range(0, inputs_flat.size(0), self.chunk_size):
            parts.append(nearest(inputs_flat[start:start + self.chunk_size]))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _ema_update(self, inputs_flat, encoding_indices):
        counts = torch.zeros(self.num_embeddings, device=inputs_flat.device, dtype=inputs_flat.dtype)
        counts.scatter_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=inputs_flat.dtype))

        sums = torch.zeros(self.num_embeddings, self.embedding_dim, device=inputs_flat.device, dtype=inputs_flat.dtype)
        sums.index_add_(0, encoding_indices, inputs_flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_w.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)

        total_count = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + self.ema_eps)
            / (total_count + self.num_embeddings * self.ema_eps)
            * total_count.clamp_min(self.ema_eps)
        )
        normalized = self.ema_w / smoothed.unsqueeze(1).clamp_min(self.ema_eps)
        self.embedding.weight.data.copy_(normalized)

        if self.dead_code_refresh and inputs_flat.numel() > 0:
            dead = torch.nonzero(self.ema_cluster_size < self.dead_code_threshold, as_tuple=False).flatten()
            if dead.numel() > 0:
                refresh_count = min(dead.numel(), max(0, self.max_code_refresh), inputs_flat.size(0))
                if refresh_count > 0:
                    dead = dead[:refresh_count]
                    source = torch.randint(0, inputs_flat.size(0), (refresh_count,), device=inputs_flat.device)
                    fresh = inputs_flat[source]
                    self.embedding.weight.data[dead] = fresh
                    self.ema_w[dead] = fresh
                    self.ema_cluster_size[dead] = self.dead_code_threshold

    def forward(self, inputs):
        # inputs: (B, C, T, H, W)
        # Flatten to (B*T*H*W, C)
        inputs_flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        
        # Find closest codebook token
        encoding_indices = self._nearest_embedding_indices(inputs_flat)

        if self.training and self.use_ema:
            self._ema_update(inputs_flat.detach(), encoding_indices)
        
        # Direct dictionary lookup completely skips allocating dense O(N*M) one-hot vectors
        # and eliminates all O(N*M*D) dense matrix dot-products overhead.
        quantized = self.embedding(encoding_indices).view(
            inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4], self.embedding_dim
        ).permute(0, 4, 1, 2, 3)
        
        # Loss: commitment loss plus codebook loss when embeddings are gradient-trained.
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        if self.use_ema:
            loss = self.commitment_cost * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        indices = encoding_indices.view(inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4])
        return quantized, loss, indices

class VideoVAE(nn.Module):
    """
    A 3D Vector-Quantized Variational Autoencoder (VQ-VAE) for latent video representation.
    Designed for consumer hardware: small channel counts and efficient causal blocks.
    """
    def __init__(self, config):
        super().__init__()
        # Extract from configs/video_config.ini style dict or parser
        in_channels = 3
        latent_channels = int(config.get('latent_channels', 4))
        base_channels = int(config.get('base_channels', 32))
        codebook_size = int(config.get('codebook_size', 4096))
        quantizer_chunk_size = int(config.get('quantizer_chunk_size', 2048))
        commitment_cost = float(config.get('commitment_cost', 0.25))
        res_blocks = max(1, int(config.get('res_blocks_per_stage', 1)))
        residual_scale = float(config.get('residual_scale', 1.0))
        use_ema = str(config.get('use_ema_quantizer', 'True')).strip().lower() in ('1', 'true', 'yes', 'on')
        ema_decay = float(config.get('ema_decay', 0.99))
        ema_eps = float(config.get('ema_eps', 1e-5))
        dead_code_refresh = str(config.get('dead_code_refresh', 'False')).strip().lower() in ('1', 'true', 'yes', 'on')
        dead_code_threshold = float(config.get('dead_code_threshold', 0.01))
        max_code_refresh = int(config.get('max_code_refresh', 64))
        spatial_downsample = int(config.get('spatial_downsample', 8))
        temporal_downsample = int(config.get('temporal_downsample', 2))
        if base_channels % 8 != 0:
            raise ValueError("VAE base_channels must be divisible by 8 for GroupNorm.")
        if spatial_downsample != 8 or temporal_downsample != 2:
            raise ValueError(
                "Current VideoVAE architecture supports spatial_downsample=8 and temporal_downsample=2 only. "
                f"Current: spatial_downsample={spatial_downsample}, temporal_downsample={temporal_downsample}."
            )

        def blocks(channels):
            return [ResnetBlock3d(channels, channels, residual_scale=residual_scale) for _ in range(res_blocks)]
        
        # Encoder: (B, 3, T, H, W) -> (B, latent_channels, T/t_ds, H/s_ds, W/s_ds)
        self.encoder = nn.Sequential(
            CausalConv3d(in_channels, base_channels, 3, stride=(1, 2, 2)), # (B, 32, T, H/2, W/2)
            *blocks(base_channels),
            CausalConv3d(base_channels, base_channels * 2, 3, stride=(2, 2, 2)), # (B, 64, T/2, H/4, W/4)
            *blocks(base_channels * 2),
            CausalConv3d(base_channels * 2, base_channels * 4, 3, stride=(1, 2, 2)), # (B, 128, T/2, H/8, W/8)
            *blocks(base_channels * 4),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, latent_channels, 3, padding=1) # (B, latent, T/2, H/8, W/8)
        )
        
        self.quantizer = VectorQuantizer(
            codebook_size,
            latent_channels,
            commitment_cost=commitment_cost,
            chunk_size=quantizer_chunk_size,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            dead_code_refresh=dead_code_refresh,
            dead_code_threshold=dead_code_threshold,
            max_code_refresh=max_code_refresh,
        )

        # Decoder: (B, latent_channels, T/2, H/8, W/8) -> (B, 3, T, H, W)
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * 4, 3, padding=1),
            *blocks(base_channels * 4),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            CausalConv3d(base_channels * 4, base_channels * 2, 3), # (B, 64, T/2, H/4, W/4)
            *blocks(base_channels * 2),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            CausalConv3d(base_channels * 2, base_channels, 3), # (B, 32, T, H/2, W/2)
            *blocks(base_channels),
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            CausalConv3d(base_channels, base_channels, 3), # (B, 32, T, H, W)
            *blocks(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(h)
        return quantized, vq_loss, indices

    def decode(self, quant):
        return self.decoder(quant)

    def forward(self, x):
        quantized, vq_loss, indices = self.encode(x)
        recon = self.decode(quantized)
        return recon, vq_loss, indices
