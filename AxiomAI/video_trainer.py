import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import configparser
import os
import random
import time
import json
import numpy as np

from video_model.vae import VideoVAE
from video_model.text_encoder import TextEncoder
from video_model.video_model import MultiScaleVideoModel
from video_generation.video_dataset import VideoDataset
from overfit_monitor import OverfitMonitor

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_param_count(n):
    if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000: return f"{n / 1_000:.1f}K"
    return str(n)

def spaced_title(text):
    return " ".join(text.upper())

def print_training_banner(title, rows, width=54):
    print("╔" + "═" * width + "╗")
    print("║" + spaced_title(title).center(width) + "║")
    print("╚" + "═" * width + "╝")
    print()
    for label, value in rows:
        print(f"  {label:<12} {value}")
    print("─" * (width + 2))

def make_video_collate(pad_id):
    def collate(batch):
        videos, captions = zip(*batch)
        videos = torch.stack(videos, dim=0)
        max_len = max(1, max(c.numel() for c in captions))
        caption_batch = torch.full((len(captions), max_len), pad_id, dtype=torch.long)
        for i, caption in enumerate(captions):
            if caption.numel() > 0:
                caption_batch[i, :caption.numel()] = caption
        return videos, caption_batch
    return collate

def split_video_dataset(dataset, val_percent, seed=1337):
    val_percent = max(0.0, min(0.95, float(val_percent)))
    val_len = max(1, int(len(dataset) * val_percent)) if len(dataset) > 1 and val_percent > 0 else 0
    if val_len >= len(dataset):
        val_len = len(dataset) - 1
    train_len = len(dataset) - val_len

    if val_len <= 0:
        return dataset, None, train_len, val_len

    generator = torch.Generator().manual_seed(int(seed))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len], generator=generator
    )
    return train_dataset, val_dataset, train_len, val_len

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

def load_state_dict_flexible(model, checkpoint_path, key=None, device='cpu'):
    payload = torch.load(checkpoint_path, map_location=device)
    if key is not None and isinstance(payload, dict) and key in payload:
        state_dict = payload[key]
    elif isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
    else:
        state_dict = payload
    result = model.load_state_dict(state_dict, strict=False)
    allowed = {'quantizer.ema_cluster_size', 'quantizer.ema_w'}
    bad_missing = [key for key in result.missing_keys if key not in allowed]
    bad_unexpected = [key for key in result.unexpected_keys if key not in allowed]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint key mismatch. Missing: {bad_missing[:8]}, unexpected: {bad_unexpected[:8]}")
    return payload

def checkpoint_metadata(train_cfg, data_cfg, vae_cfg, v_cfg=None, extra=None):
    meta = {
        'width': int(train_cfg.get('width', 64)),
        'height': int(train_cfg.get('height', 64)),
        'fps': float(data_cfg.get('fps', 8.0)),
        'duration': float(data_cfg.get('duration', 2.0)),
        'codebook_size': int(vae_cfg.get('codebook_size', 4096)),
        'base_channels': int(vae_cfg.get('base_channels', 32)),
        'latent_channels': int(vae_cfg.get('latent_channels', 4)),
        'res_blocks_per_stage': int(vae_cfg.get('res_blocks_per_stage', 1)),
        'use_ema_quantizer': str(vae_cfg.get('use_ema_quantizer', 'True')).strip().lower() in ('1', 'true', 'yes', 'on'),
        'spatial_downsample': int(vae_cfg.get('spatial_downsample', 8)),
        'temporal_downsample': int(vae_cfg.get('temporal_downsample', 2)),
    }
    if v_cfg is not None:
        meta['video_model'] = dict(v_cfg)
    if extra:
        meta.update(extra)
    return meta

def psnr_from_mse(mse):
    mse = max(float(mse), 1e-12)
    return 10.0 * np.log10(4.0 / mse)

def codebook_usage(indices, codebook_size):
    used = torch.unique(indices.detach()).numel()
    return used, used / max(1, int(codebook_size))

def video_to_uint8(video):
    arr = video.detach().cpu().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).byte()
    return arr

def save_recon_preview(original, recon, out_dir, epoch, max_frames=6):
    try:
        import cv2
    except ImportError:
        return None
    os.makedirs(out_dir, exist_ok=True)
    orig = video_to_uint8(original[0]).permute(1, 2, 3, 0).numpy()
    rec = video_to_uint8(recon[0]).permute(1, 2, 3, 0).numpy()
    frame_count = min(max_frames, orig.shape[0])
    if frame_count <= 0:
        return None
    picks = np.linspace(0, orig.shape[0] - 1, frame_count).astype(np.int64)
    strips = []
    for idx in picks:
        pair = np.concatenate([orig[idx], rec[idx]], axis=0)
        strips.append(pair)
    strip = np.concatenate(strips, axis=1)
    strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(out_dir, f"vae_recon_epoch_{epoch:04d}.png")
    cv2.imwrite(out_path, strip)
    return out_path

def cfg_float(config, key, default):
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return float(default)

def vae_reconstruction_loss(recon, target, vq_loss, train_cfg):
    l1 = F.l1_loss(recon, target)
    mse = F.mse_loss(recon, target)
    loss = (
        cfg_float(train_cfg, 'vae_l1_weight', 1.0) * l1
        + cfg_float(train_cfg, 'vae_mse_weight', 0.5) * mse
        + cfg_float(train_cfg, 'vae_vq_weight', 1.0) * vq_loss.mean()
    )

    temporal = recon.new_tensor(0.0)
    if recon.size(2) > 1:
        temporal = F.l1_loss(recon[:, :, 1:] - recon[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
        loss = loss + cfg_float(train_cfg, 'vae_temporal_weight', 0.1) * temporal

    spatial_h = recon.new_tensor(0.0)
    spatial_w = recon.new_tensor(0.0)
    if recon.size(3) > 1:
        spatial_h = F.l1_loss(recon[:, :, :, 1:] - recon[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
    if recon.size(4) > 1:
        spatial_w = F.l1_loss(recon[:, :, :, :, 1:] - recon[:, :, :, :, :-1], target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
    spatial = 0.5 * (spatial_h + spatial_w)
    loss = loss + cfg_float(train_cfg, 'vae_spatial_weight', 0.05) * spatial

    return loss, {
        'l1': l1.detach(),
        'mse': mse.detach(),
        'temporal': temporal.detach(),
        'spatial': spatial.detach(),
        'vq': vq_loss.detach().mean(),
    }

def preflight_video_dataset(dataset, strict=True):
    problems = dataset.validate_files()
    fatal_markers = ("could not open", "no readable frames", "caption is empty", "caption read failed")
    fatal = [p for p in problems if any(marker in p.lower() for marker in fatal_markers)]

    if problems:
        print("  Video data preflight:")
        for problem in problems:
            label = "FATAL" if problem in fatal else "WARN"
            print(f"    [{label}] {problem}")

    if strict and fatal:
        print("FATAL ERROR: Video preflight failed. Fix the files above or set strict_video_loading = False.")
        return False
    return True

class LatentVideoDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['coarse'], sample['fine'], sample['caption']

def make_latent_collate(pad_id):
    def collate(batch):
        coarse, fine, captions = zip(*batch)
        coarse = torch.stack(coarse, dim=0).long()
        fine = torch.stack(fine, dim=0).long()
        max_len = max(1, max(c.numel() for c in captions))
        caption_batch = torch.full((len(captions), max_len), pad_id, dtype=torch.long)
        for i, caption in enumerate(captions):
            if caption.numel() > 0:
                caption_batch[i, :caption.numel()] = caption.long()
        return coarse, fine, caption_batch
    return collate

def latent_cache_metadata(dataset, train_cfg, data_cfg, vae_cfg):
    vae_path = train_cfg.get('vae_checkpoint_path', 'model/video_model/vae_checkpoint.pth')
    vae_mtime = os.path.getmtime(vae_path) if os.path.exists(vae_path) else 0.0
    return checkpoint_metadata(train_cfg, data_cfg, vae_cfg, extra={
        'video_files': list(dataset.video_files),
        'vae_checkpoint_path': vae_path,
        'vae_checkpoint_mtime': vae_mtime,
        'max_caption_len': dataset.max_caption_len,
    })

def load_latent_cache(cache_path, expected_meta):
    if not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location='cpu')
    except Exception as e:
        print(f"  [Cache] Could not read latent cache: {e}")
        return None
    if not isinstance(payload, dict) or 'metadata' not in payload or 'samples' not in payload:
        print("  [Cache] Existing latent cache is from an old format; rebuilding.")
        return None
    if payload['metadata'] != expected_meta:
        print("  [Cache] Metadata changed; rebuilding latent cache.")
        return None
    return payload['samples']

def build_latent_cache(vae, dataset, train_cfg, data_cfg, vae_cfg, device):
    cache_path = train_cfg.get('latent_cache_path', 'data/processed/video_latent_cache.pt')
    rebuild = train_cfg.getboolean('rebuild_latent_cache', fallback=False)
    metadata = latent_cache_metadata(dataset, train_cfg, data_cfg, vae_cfg)

    if not rebuild:
        samples = load_latent_cache(cache_path, metadata)
        if samples is not None:
            print(f"  [Cache] Loaded {len(samples)} VAE latent samples from {cache_path}")
            return samples

    print("  [Cache] Building VAE latent cache. This runs the VAE once, then AR training skips video decoding.")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    samples = []
    old_flip = dataset.flip_prob
    old_crop = dataset.crop_pad_percent
    dataset.flip_prob = 0.0
    dataset.crop_pad_percent = 0.0
    vae.eval()
    try:
        with torch.no_grad():
            for idx in range(len(dataset)):
                video, caption = dataset[idx]
                video = video.unsqueeze(0).to(device)
                scaled_video = F.interpolate(video, scale_factor=(1.0, 0.5, 0.5), mode='trilinear', align_corners=False)
                _, _, coarse = vae.encode(scaled_video)
                _, _, fine = vae.encode(video)
                samples.append({
                    'coarse': coarse.squeeze(0).cpu().int(),
                    'fine': fine.squeeze(0).cpu().int(),
                    'caption': caption.cpu().long(),
                })
                if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
                    print(f"    Cached {idx + 1}/{len(dataset)} clips")
    finally:
        dataset.flip_prob = old_flip
        dataset.crop_pad_percent = old_crop

    torch.save({'metadata': metadata, 'samples': samples}, cache_path)
    print(f"  [Cache] Saved latent cache to {cache_path}")
    return samples

def validate_vae_checkpoint_quality(payload, train_cfg):
    if not train_cfg.getboolean('enforce_vae_quality_gate', fallback=False):
        return True
    if not isinstance(payload, dict) or 'metadata' not in payload:
        print("FATAL ERROR: VAE checkpoint has no quality metadata. Retrain the VAE with the current trainer before AR training.")
        return False

    meta = payload.get('metadata', {})
    val_psnr = meta.get('val_psnr')
    code_usage = meta.get('val_codebook_usage')
    min_psnr = cfg_float(train_cfg, 'min_vae_psnr_for_ar', 16.0)
    min_usage = cfg_float(train_cfg, 'min_vae_code_usage_for_ar', 0.005)

    if val_psnr is None or code_usage is None:
        print("FATAL ERROR: VAE checkpoint is missing PSNR/codebook quality metadata. Retrain the VAE.")
        return False
    if float(val_psnr) < min_psnr:
        print(f"FATAL ERROR: VAE quality gate failed. PSNR {float(val_psnr):.2f}dB < required {min_psnr:.2f}dB.")
        return False
    if float(code_usage) < min_usage:
        print(f"FATAL ERROR: VAE quality gate failed. Codebook usage {float(code_usage)*100:.2f}% < required {min_usage*100:.2f}%.")
        return False
    return True

def validate_video_setup(v_cfg, t_cfg, vae_cfg, train_cfg, data_cfg):
    d_model = int(v_cfg.get('d_model', 128))
    n_heads = int(v_cfg.get('n_heads', 8))
    if d_model % n_heads != 0:
        raise ValueError(f"VIDEO_MODEL d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

    head_dim = d_model // n_heads
    d_t = head_dim // 4
    d_h = (head_dim - d_t) // 2
    d_w = head_dim - d_t - d_h
    if min(d_t, d_h, d_w) < 2 or d_t % 2 or d_h % 2 or d_w % 2:
        raise ValueError(f"3D RoPE needs even non-empty temporal/spatial splits. head_dim={head_dim} gives {(d_t, d_h, d_w)}.")

    width = int(train_cfg.get('width', 64))
    height = int(train_cfg.get('height', 64))
    fps = float(data_cfg.get('fps', 8.0))
    duration = float(data_cfg.get('duration', 2.0))
    frames = int(fps * duration)
    spatial_ds = int(vae_cfg.get('spatial_downsample', 8))
    temporal_ds = int(vae_cfg.get('temporal_downsample', 2))
    max_seq = int(v_cfg.get('max_seq_len', 2048))
    codebook_size = int(vae_cfg.get('codebook_size', 4096))
    latent_channels = int(vae_cfg.get('latent_channels', 4))
    base_channels = int(vae_cfg.get('base_channels', 32))
    quantizer_chunk = int(vae_cfg.get('quantizer_chunk_size', 2048))
    bos_id = int(v_cfg.get('bos_id', codebook_size))
    batch_size = int(train_cfg.get('batch_size', 1))
    epochs = int(train_cfg.get('epochs', 1))

    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("Video width, height, and frame count must all be positive.")
    if spatial_ds < 1 or temporal_ds < 1:
        raise ValueError("VAE spatial_downsample and temporal_downsample must both be >= 1.")
    if spatial_ds != 8 or temporal_ds != 2:
        raise ValueError(
            "Current VideoVAE architecture supports spatial_downsample=8 and temporal_downsample=2 only. "
            f"Current: spatial_downsample={spatial_ds}, temporal_downsample={temporal_ds}."
        )
    if codebook_size < 2:
        raise ValueError("VAE codebook_size must be >= 2.")
    if latent_channels < 1:
        raise ValueError("VAE latent_channels must be >= 1.")
    if base_channels < 8 or base_channels % 8 != 0:
        raise ValueError("VAE base_channels must be >= 8 and divisible by 8 for GroupNorm.")
    if quantizer_chunk < 0:
        raise ValueError("VAE quantizer_chunk_size must be >= 0.")
    if batch_size < 1 or epochs < 1:
        raise ValueError("Video batch_size and epochs must both be >= 1.")
    if width % spatial_ds != 0 or height % spatial_ds != 0:
        raise ValueError(f"Video width/height must be divisible by spatial_downsample={spatial_ds}. Current: {width}x{height}.")
    coarse_factor = spatial_ds * 2
    if width % coarse_factor != 0 or height % coarse_factor != 0:
        raise ValueError(
            f"Video width/height must be divisible by {coarse_factor} for the two-scale video path. "
            f"Current: {width}x{height}, spatial_downsample={spatial_ds}."
        )
    if frames % temporal_ds != 0:
        raise ValueError(f"fps * duration ({frames}) must be divisible by temporal_downsample={temporal_ds}.")
    if (width // spatial_ds) < 2 or (height // spatial_ds) < 2:
        raise ValueError("Latent spatial grid is too small for the two-scale video generator.")
    if max(frames // temporal_ds, width // spatial_ds, height // spatial_ds) >= max_seq:
        raise ValueError("VIDEO_MODEL max_seq_len must be larger than the largest latent coordinate.")
    if bos_id < codebook_size:
        raise ValueError(f"bos_id ({bos_id}) must be >= codebook_size ({codebook_size}).")
    if int(t_cfg.get('max_seq_len', 64)) < 1:
        raise ValueError("TEXT_ENCODER max_seq_len must be positive.")

def load_setup():
    config = configparser.ConfigParser()
    config.read('configs/video_config.ini')
    
    v_cfg = config['VIDEO_MODEL']
    t_cfg = config['TEXT_ENCODER']
    vae_cfg = config['VAE']
    train_cfg = config['TRAINING']
    data_cfg = config['DATA']
    
    # Enforce textual dimensions universally so configurations cannot drift independently
    base_cfg = configparser.ConfigParser()
    base_cfg.read('configs/config.ini')
    if base_cfg.has_section('MODEL'):
        t_cfg['vocab_size'] = base_cfg['MODEL'].get('vocab_size', t_cfg.get('vocab_size', '2096'))
        t_cfg['max_seq_len'] = base_cfg['MODEL'].get('max_seq_len', t_cfg.get('max_seq_len', '64'))
    
    data_cfg['width'] = train_cfg.get('width', data_cfg.get('width', '64'))
    data_cfg['height'] = train_cfg.get('height', data_cfg.get('height', '64'))
    data_cfg['strict_video_loading'] = train_cfg.get('strict_video_loading', data_cfg.get('strict_video_loading', 'True'))
    v_cfg['codebook_size'] = vae_cfg.get('codebook_size', v_cfg.get('codebook_size', '4096'))
    v_cfg['bos_id'] = v_cfg.get('bos_id', v_cfg['codebook_size'])
    if int(v_cfg['bos_id']) != int(v_cfg['codebook_size']):
        print(f"  [Video Config] Aligning bos_id to codebook_size ({v_cfg['codebook_size']}) for a clean VQ token boundary.")
        v_cfg['bos_id'] = v_cfg['codebook_size']
    validate_video_setup(v_cfg, t_cfg, vae_cfg, train_cfg, data_cfg)
    
    device = train_cfg.get('device', 'cpu')
    if device == 'auto' or device == 'cuda':
        device = 'cpu'
        
    num_threads = int(train_cfg.get('num_threads', 0))
    safe_set_torch_threads(num_threads)
        
    vocab_path = configparser.ConfigParser()
    vocab_path.read('configs/config.ini')
    tokenizer_path = vocab_path['DATA']['vocab_path']
    
    return config, v_cfg, t_cfg, vae_cfg, train_cfg, data_cfg, device, tokenizer_path

def train_vae():
    _, _, t_cfg, vae_cfg, train_cfg, data_cfg, device, tokenizer_path = load_setup()
    
    vae = VideoVAE(vae_cfg).to(device)
    vae_payload = None
    
    dataset = VideoDataset(data_cfg, tokenizer_path, width=train_cfg.get('width'), height=train_cfg.get('height'), max_caption_len=t_cfg.get('max_seq_len', 64))
    if len(dataset) == 0:
        print("FATAL ERROR: No validated video-text pairs found.")
        return
    strict_video = train_cfg.getboolean('strict_video_loading', fallback=True)
    if not preflight_video_dataset(dataset, strict_video):
        return
    batch_size = int(train_cfg['batch_size'])
    val_percent = float(train_cfg.get('val_split', 0.2))
    split_seed = int(train_cfg.get('split_seed', 1337))
    train_dataset, val_dataset, train_len, val_len = split_video_dataset(dataset, val_percent, split_seed)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=make_video_collate(dataset.tokenizer.pad_id))
    else:
        val_loader = None

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_video_collate(dataset.tokenizer.pad_id))
    
    lr = float(train_cfg['lr'])
    optimizer = optim.AdamW(vae.parameters(), lr=lr)
    
    epochs = int(train_cfg['epochs'])
    total_steps = max(1, epochs * max(1, len(dataloader)))
    scheduler = None
    if train_cfg.getboolean('vae_cosine_schedule', fallback=True):
        min_lr = lr * cfg_float(train_cfg, 'vae_min_lr_ratio', 0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    best_loss = float('inf')
    overfit_monitor = OverfitMonitor("Video VAE", gap_threshold=0.08)
    
    num_params = sum(p.numel() for p in vae.parameters())
    codebook = int(vae_cfg.get('codebook_size', 4096))
    latent_ch = int(vae_cfg.get('latent_channels', 4))
    fps = float(data_cfg.get('fps', 8.0))
    duration = float(data_cfg.get('duration', 2.0))
    
    clear_screen()
    print("╔══════════════════════════════════════════════╗")
    print("║         V A E   T R A I N I N G   R U N      ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print(f"  🧠 VAE         {format_param_count(num_params)} params")
    print(f"  📐 Latent      {latent_ch}ch  │  Codebook: {codebook} tokens")
    print(f"  ⚙  Config      {device.upper()}  │  Batch: {batch_size}  │  LR: {lr}")
    print(f"  📊 Data        {len(dataset)} video clips  │  {duration}s @ {fps:.0f}fps")
    print(f"  🎯 Epochs      {epochs}")
    print("─" * 48)
    
    clear_screen()
    print_training_banner("VAE Training Run", [
        ("🧠 Model", f"{format_param_count(num_params)} params  │  VQ-VAE"),
        ("📐 Latent", f"{latent_ch}ch  │  Codebook: {codebook} tokens"),
        ("⚙ Config", f"{device.upper()}  │  Batch: {batch_size}  │  LR: {lr}"),
        ("📊 Data", f"Train: {train_len}  │  Val: {val_len}  │  {duration}s @ {fps:.0f}fps"),
        ("💾 Save", train_cfg['vae_checkpoint_path']),
    ])

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0
        train_recon_sum = 0.0
        train_batches = 0
        train_used_codes = set()
        
        for i, (video, _) in enumerate(dataloader):
            video = video.to(device)
            
            recon, vq_loss, indices = vae(video)
            loss, loss_parts = vae_reconstruction_loss(recon, video, vq_loss, train_cfg)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg_float(train_cfg, 'vae_grad_clip', 1.0))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            train_recon_sum += loss_parts['mse'].item()
            train_batches += 1
            train_used_codes.update(torch.unique(indices.detach()).cpu().tolist())
            
            if len(dataloader) > 1 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(dataloader)}] VAE Loss: {loss.item():.4f} (L1: {loss_parts['l1'].item():.4f}, MSE: {loss_parts['mse'].item():.4f}, VQ: {loss_parts['vq'].item():.4f})")
                
        avg_train_loss = epoch_loss / max(1, len(dataloader))
        avg_train_recon = train_recon_sum / max(1, train_batches)
        train_psnr = psnr_from_mse(avg_train_recon)
        train_code_usage = len(train_used_codes) / max(1, codebook)

        avg_val_loss = avg_train_loss
        val_psnr = train_psnr
        val_code_usage = train_code_usage
        preview_original = None
        preview_recon = None

        if val_loader:
            vae.eval()
            val_loss_sum = 0
            val_recon_sum = 0.0
            val_batches = 0
            val_used_codes = set()
            old_flip = dataset.flip_prob
            old_crop = dataset.crop_pad_percent
            dataset.flip_prob = 0.0
            dataset.crop_pad_percent = 0.0
            try:
                with torch.no_grad():
                    for video, _ in val_loader:
                        video = video.to(device)
                        recon, vq_loss, indices = vae(video)
                        loss, loss_parts = vae_reconstruction_loss(recon, video, vq_loss, train_cfg)
                        val_loss_sum += loss.item()
                        val_recon_sum += loss_parts['mse'].item()
                        val_batches += 1
                        val_used_codes.update(torch.unique(indices.detach()).cpu().tolist())
                        if preview_original is None:
                            preview_original = video.detach().cpu()
                            preview_recon = recon.detach().cpu()
            finally:
                dataset.flip_prob = old_flip
                dataset.crop_pad_percent = old_crop
            avg_val_loss = val_loss_sum / max(1, val_batches)
            avg_val_recon = val_recon_sum / max(1, val_batches)
            val_psnr = psnr_from_mse(avg_val_recon)
            val_code_usage = len(val_used_codes) / max(1, codebook)

        # Checkpoint logically secured directly behind mathematical validation improvements
        tag = ""
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(os.path.dirname(train_cfg['vae_checkpoint_path']), exist_ok=True)
            torch.save({
                'state_dict': vae.state_dict(),
                'epoch': epoch,
                'best_val_loss': avg_val_loss,
                'metadata': checkpoint_metadata(train_cfg, data_cfg, vae_cfg, extra={
                    'train_clips': train_len,
                    'val_clips': val_len,
                    'val_psnr': float(val_psnr),
                    'val_codebook_usage': float(val_code_usage),
                    'loss_weights': {
                        'l1': cfg_float(train_cfg, 'vae_l1_weight', 1.0),
                        'mse': cfg_float(train_cfg, 'vae_mse_weight', 0.5),
                        'vq': cfg_float(train_cfg, 'vae_vq_weight', 1.0),
                        'temporal': cfg_float(train_cfg, 'vae_temporal_weight', 0.1),
                        'spatial': cfg_float(train_cfg, 'vae_spatial_weight', 0.05),
                    },
                })
            }, train_cfg['vae_checkpoint_path'])
            tag = "  ✨ New Best"

        print(f"  Epoch {epoch+1:<3} │ T Loss {avg_train_loss:.4f}  │  V Loss {avg_val_loss:.4f}{tag}")
        print(f"    Quality: PSNR {val_psnr:.2f}dB | Codebook used {val_code_usage*100:.1f}%")
        preview_path = None
        save_preview = train_cfg.getboolean('save_vae_recon_preview', fallback=True)
        preview_every = max(1, int(train_cfg.get('preview_every_epochs', 5)))
        if save_preview and preview_original is not None and (avg_val_loss <= best_loss or epoch == 0 or (epoch + 1) % preview_every == 0):
            preview_path = save_recon_preview(
                preview_original,
                preview_recon,
                train_cfg.get('preview_path', 'model/video_model/previews'),
                epoch + 1,
            )
        if preview_path:
            print(f"    Preview: {preview_path}")

        if val_loader:
            overfit_warning = overfit_monitor.update(epoch + 1, avg_train_loss, avg_val_loss)
            if overfit_warning:
                print(overfit_warning)

def train_text_encoder():
    config, v_cfg, t_cfg, vae_cfg, train_cfg, data_cfg, device, tokenizer_path = load_setup()
    
    data_path = data_cfg.get('video_data_path', 'data/Videos/')
    if not os.path.exists(data_path):
        print(f"❌ FATAL ERROR: Video directory {data_path} not found.")
        return
        
    txt_files = [f for f in os.listdir(data_path) if f.lower().endswith('.txt')]
    if len(txt_files) == 0:
        print(f"❌ FATAL ERROR: No .txt caption files found in {data_path}.")
        return
        
    from tokenizer.my_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    if not tokenizer.load(tokenizer_path):
        print(f"❌ FATAL ERROR: Failed to load tokenizer from {tokenizer_path}")
        return
        
    all_tokens = []
    for tf in txt_files:
        with open(os.path.join(data_path, tf), 'r', encoding='utf-8') as f:
            tokens = tokenizer.encode(f.read().strip())
            all_tokens.extend(tokens + [tokenizer.pad_id])
            
    data = np.array(all_tokens, dtype=np.int64)
    max_seq_len = int(t_cfg.get('max_seq_len', 64))
    
    if len(data) <= max_seq_len:
        print("❌ FATAL ERROR: Not enough text data (fewer tokens than max_seq_len). Add more videos.")
        return
        
    text_encoder = TextEncoder(t_cfg).to(device)
    
    # Create language modeling head
    vocab_size = int(t_cfg.get('vocab_size', 2096))
    d_model = int(t_cfg.get('d_model', 128))
    lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device)
    # Structure Hardening: Weight tying - explicitly share weights with embedding layer
    lm_head.weight = text_encoder.tok_emb.weight
    
    lr = float(train_cfg['lr'])
    optimizer = optim.AdamW(list(text_encoder.parameters()), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    
    batch_size = int(train_cfg.get('text_batch_size', int(train_cfg['batch_size']) * 4))
    epochs = int(train_cfg['epochs'])
    best_loss = float('inf')
    
    num_params = sum(p.numel() for p in text_encoder.parameters())
    n_layers = int(t_cfg.get('n_layers', 2))
    n_heads = int(t_cfg.get('n_heads', 4))
    
    clear_screen()
    print("╔══════════════════════════════════════════════╗")
    print("║     T E X T   E N C O D E R   T R A I N     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print(f"  🧠 Encoder     {format_param_count(num_params)} params  │  {n_layers}L {n_heads}H d={d_model}")
    print(f"  ⚙  Config      {device.upper()}  │  Batch: {batch_size}  │  LR: {lr}")
    print(f"  📊 Data        {len(txt_files)} captions  │  {len(data):,} tokens  │  Seq: {max_seq_len}")
    print(f"  🎯 Epochs      {epochs}")
    print("─" * 48)
    
    clear_screen()
    print_training_banner("Text Encoder Train", [
        ("🧠 Model", f"{format_param_count(num_params)} params  │  {n_layers}L {n_heads}H d={d_model}"),
        ("⚙ Config", f"{device.upper()}  │  Batch: {batch_size}  │  LR: {lr}"),
        ("📊 Data", f"{len(txt_files)} captions  │  {len(data):,} tokens  │  Seq: {max_seq_len}"),
        ("💾 Save", train_cfg.get('text_encoder_checkpoint_path', 'model/video_model/text_encoder_checkpoint.pth')),
    ])

    class VideoTextDataset(torch.utils.data.Dataset):
        def __init__(self, data_arr, seq_len):
            self.data = data_arr
            self.seq_len = seq_len
        def __len__(self):
            return max(0, len(self.data) - self.seq_len - 1)
        def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len].astype(np.int64)
            y = self.data[idx+1:idx+1+self.seq_len].astype(np.int64)
            return torch.from_numpy(x), torch.from_numpy(y)

    dataset = VideoTextDataset(data, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        text_encoder.train()
        epoch_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # For standalone LM, we MUST enforce causal masking
            h = text_encoder(x, is_causal=True)
            logits = lm_head(h)
            
            loss = ce_loss(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if len(dataloader) > 1 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / max(1, len(dataloader))
        tag = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = train_cfg.get('text_encoder_checkpoint_path', 'model/video_model/text_encoder_checkpoint.pth')
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(text_encoder.state_dict(), ckpt_path)
            tag = "  ✨ New Best"

        print(f"  Epoch {epoch+1:<3} │ T Loss {avg_loss:.4f}{tag}")


def train_video_model():
    config, v_cfg, t_cfg, vae_cfg, train_cfg, data_cfg, device, tokenizer_path = load_setup()

    vae = VideoVAE(vae_cfg).to(device)
    vae_payload = None
    vae_status = "❌ Random (Untrained!)"
    if os.path.exists(train_cfg['vae_checkpoint_path']):
        vae_payload = load_state_dict_flexible(vae, train_cfg['vae_checkpoint_path'], key='state_dict', device=device)
        vae_status = "✅ Pre-Trained"
    if not os.path.exists(train_cfg['vae_checkpoint_path']):
        print(f"FATAL ERROR: VAE checkpoint not found at {train_cfg['vae_checkpoint_path']}. Train the Video VAE before Video AR.")
        return
    if not validate_vae_checkpoint_quality(vae_payload, train_cfg):
        return
    vae.eval() # Freeze VAE during AR training
    
    text_encoder = TextEncoder(t_cfg).to(device)
    te_status = "❌ Random"
    text_ckpt = train_cfg.get('text_encoder_checkpoint_path', 'model/video_model/text_encoder_checkpoint.pth')
    if os.path.exists(text_ckpt):
        text_encoder.load_state_dict(torch.load(text_ckpt, map_location=device))
        te_status = "✅ Pre-Trained"
    video_model = MultiScaleVideoModel(v_cfg).to(device)
    
    codebook_size = int(vae_cfg.get('codebook_size', 4096))
    fine_tune_text_encoder = train_cfg.getboolean('fine_tune_text_encoder', fallback=False)
    if not fine_tune_text_encoder:
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)
    
    dataset = VideoDataset(data_cfg, tokenizer_path, width=train_cfg.get('width'), height=train_cfg.get('height'), max_caption_len=t_cfg.get('max_seq_len', 64))
    if len(dataset) == 0:
        print("FATAL ERROR: No validated video-text pairs found.")
        return
    strict_video = train_cfg.getboolean('strict_video_loading', fallback=True)
    if not preflight_video_dataset(dataset, strict_video):
        return
    
    val_percent = float(train_cfg.get('val_split', 0.2))
    split_seed = int(train_cfg.get('split_seed', 1337))
    batch_size = int(train_cfg['batch_size'])
    use_cached_latents = train_cfg.getboolean('use_latent_cache', fallback=False) and os.path.exists(train_cfg['vae_checkpoint_path'])

    if use_cached_latents:
        latent_samples = build_latent_cache(vae, dataset, train_cfg, data_cfg, vae_cfg, device)
        latent_dataset = LatentVideoDataset(latent_samples)
        train_dataset, val_dataset, train_len, val_len = split_video_dataset(latent_dataset, val_percent, split_seed)
        collate_fn = make_latent_collate(dataset.tokenizer.pad_id)
    else:
        train_dataset, val_dataset, train_len, val_len = split_video_dataset(dataset, val_percent, split_seed)
        collate_fn = make_video_collate(dataset.tokenizer.pad_id)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    lr = float(train_cfg['lr'])
    trainable_params = list(video_model.parameters())
    if fine_tune_text_encoder:
        trainable_params += list(text_encoder.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    epochs = int(train_cfg['epochs'])
    best_loss = float('inf')
    overfit_monitor = OverfitMonitor("Video AR", gap_threshold=0.45)
    
    ar_params = sum(p.numel() for p in video_model.parameters())
    te_params = sum(p.numel() for p in text_encoder.parameters())
    d_model = int(v_cfg.get('d_model', 128))
    n_layers = int(v_cfg.get('n_layers', 4))
    n_heads = int(v_cfg.get('n_heads', 8))
    fps = float(data_cfg.get('fps', 8.0))
    duration = float(data_cfg.get('duration', 2.0))
    
    clear_screen()
    print("╔══════════════════════════════════════════════╗")
    print("║      V I D E O   A R   T R A I N I N G       ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print(f"  🧠 AR Model    {format_param_count(ar_params)} params  │  {n_layers}L {n_heads}H d={d_model}")
    print(f"  📝 Text Enc    {format_param_count(te_params)} params  │  {te_status}")
    print(f"  🎨 VAE         {vae_status}  │  Codebook: {codebook_size}")
    print(f"  ⚙  Config      {device.upper()}  │  Batch: {batch_size}  │  LR: {lr}")
    print(f"  📊 Data        Train: {train_len}  │  Val: {val_len}  │  {duration}s @ {fps:.0f}fps")
    print(f"  🎯 Epochs      {epochs}")
    print("─" * 48)
    
    clear_screen()
    print_training_banner("Video AR Training", [
        ("🧠 Model", f"{format_param_count(ar_params)} params  │  {n_layers}L {n_heads}H d={d_model}"),
        ("📝 Text", f"{format_param_count(te_params)} params  │  {te_status}"),
        ("🎨 VAE", f"{vae_status}  │  Codebook: {codebook_size}"),
        ("⚙ Config", f"{device.upper()}  │  Batch: {batch_size}  │  LR: {lr}"),
        ("📊 Data", f"Train: {train_len}  │  Val: {val_len}  │  {duration}s @ {fps:.0f}fps"),
        ("💾 Save", train_cfg['checkpoint_path']),
    ])

    coord_cache = {}

    def get_3d_coords(batch_count, t_len, h_len, w_len):
        key = (batch_count, t_len, h_len, w_len)
        if key not in coord_cache:
            coord_cache[key] = (
                torch.arange(t_len, device=device).repeat_interleave(h_len*w_len).repeat(batch_count, 1),
                torch.arange(h_len, device=device).repeat(t_len).repeat_interleave(w_len).repeat(batch_count, 1),
                torch.arange(w_len, device=device).repeat(t_len*h_len).repeat(batch_count, 1),
            )
        return coord_cache[key]

    for epoch in range(epochs):
        video_model.train()
        text_encoder.train(fine_tune_text_encoder)
        
        epoch_loss = 0
        
        def compute_step(batch_data, caption_tokens, forced_scale_id=None):
            caption_tokens = caption_tokens.to(device)
            
            # --- Authentic Multi-Scale Construction ---
            scale_id = int(forced_scale_id) if forced_scale_id is not None else torch.randint(0, 2, (1,)).item()
            if use_cached_latents:
                coarse_indices, fine_indices = batch_data
                coarse_indices = coarse_indices.to(device).long()
                fine_indices = fine_indices.to(device).long()
                b = fine_indices.size(0)
                indices = coarse_indices if scale_id == 0 else fine_indices
            else:
                video = batch_data.to(device)
                b = video.size(0)
            scale_tensor = torch.full((b,), scale_id, device=device, dtype=torch.long)
            
            if not use_cached_latents:
                with torch.no_grad():
                    if scale_id == 0:
                        # Scale 0: Downsample video by 50% spatial logic
                        scaled_video = F.interpolate(video, scale_factor=(1.0, 0.5, 0.5), mode='trilinear', align_corners=False)
                        _, _, indices = vae.encode(scaled_video)
                    else:
                        # Scale 1: Full resolution
                        _, _, indices = vae.encode(video)
                        scaled_video = F.interpolate(video, scale_factor=(1.0, 0.5, 0.5), mode='trilinear', align_corners=False)
                        _, _, coarse_indices = vae.encode(scaled_video)

            if fine_tune_text_encoder:
                text_emb = text_encoder(caption_tokens)
            else:
                with torch.no_grad():
                    text_emb = text_encoder(caption_tokens)

            # Cross-Attention Hierarchical Cascade
            context_emb = text_emb
            if scale_id == 1:
                # Embed the coarse backbone and crush it exclusively over the timeline
                # This drops Cross-Attention footprint geometrically while preserving precise X/Y skeleton layouts!
                _, t_c, h_c, w_c = coarse_indices.shape
                coarse_emb = video_model.tok_embed(coarse_indices.view(b, -1))
                coarse_emb = coarse_emb.view(b, t_c, h_c * w_c, -1).mean(dim=1)
                context_emb = torch.cat([text_emb, coarse_emb], dim=1)

            # Flatten indices: (B, T, H, W) -> (B, T*H*W)
            indices_flat = indices.view(b, -1)
            t_len, h_len, w_len = indices.shape[1], indices.shape[2], indices.shape[3]
            frame_size = h_len * w_len
            
            # Shift sequences physically by 1 Entire Frame to align with parallel decoding!
            bos_frame = torch.full((b, frame_size), video_model.bos_id, device=device, dtype=torch.long)
            input_ids = torch.cat([bos_frame, indices_flat[:, :-frame_size]], dim=1)
            target_ids = indices_flat
            
            # Parallel Decode Coordinates:
            # We pass the full flattened shape exactly as they span the raster order.
            t_coords, h_coords, w_coords = get_3d_coords(b, t_len, h_len, w_len)

            # Parallel Forward using conditional context array
            preds, _ = video_model(input_ids, context_emb, scale_tensor, (t_len, h_len, w_len), t_coords, h_coords, w_coords)
            
            # (B, C, S)
            return scale_id, ce_loss(preds.transpose(1, 2), target_ids)

        for i, batch in enumerate(train_loader):
            if use_cached_latents:
                coarse_indices, fine_indices, caption_tokens = batch
                scale_id, ar_loss = compute_step((coarse_indices, fine_indices), caption_tokens)
            else:
                video, caption_tokens = batch
                scale_id, ar_loss = compute_step(video, caption_tokens)
            
            optimizer.zero_grad(set_to_none=True)
            ar_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            epoch_loss += ar_loss.item()
            
            if len(train_loader) > 1 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] Scale: {scale_id} AR Loss: {ar_loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        
        # Validation Loop
        avg_val_loss = avg_train_loss
        if val_loader:
            video_model.eval()
            text_encoder.eval()
            val_loss_sum = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if use_cached_latents:
                        coarse_indices, fine_indices, caption_tokens = batch
                        batch_data = (coarse_indices, fine_indices)
                    else:
                        batch_data, caption_tokens = batch
                    _, v_loss_0 = compute_step(batch_data, caption_tokens, forced_scale_id=0)
                    _, v_loss_1 = compute_step(batch_data, caption_tokens, forced_scale_id=1)
                    val_loss_sum += 0.5 * (v_loss_0.item() + v_loss_1.item())
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(1, val_batches)
            
        # Verify mathematically valid loss structures before overwriting critical arrays
        tag = ""
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(os.path.dirname(train_cfg['checkpoint_path']), exist_ok=True)
            torch.save({
                'video_model': video_model.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'vae': vae.state_dict(),
                'vae_metadata': vae_payload.get('metadata', {}) if isinstance(vae_payload, dict) else {},
                'epoch': epoch,
                'best_val_loss': best_loss,
                'config': dict(v_cfg)
            }, train_cfg['checkpoint_path'])
            tag = "  ✨ New Best"

        print(f"  Epoch {epoch+1:<3} │ T Loss {avg_train_loss:.4f}  │  V Loss {avg_val_loss:.4f}{tag}")
        if val_loader:
            overfit_warning = overfit_monitor.update(epoch + 1, avg_train_loss, avg_val_loss)
            if overfit_warning:
                print(overfit_warning)

if __name__ == "__main__":
    train_video_model()
