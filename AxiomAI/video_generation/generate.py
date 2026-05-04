import torch
import torch.nn.functional as F
import configparser
import os
import cv2
import numpy as np

from video_model.vae import VideoVAE
from video_model.text_encoder import TextEncoder
from video_model.video_model import MultiScaleVideoModel
from tokenizer.my_tokenizer import CharTokenizer

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

def sample_top_k_top_p(logits, top_k=40, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
        
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    
    # Mathematical array collapsing to natively execute 3D Multidimensional token batch calculations
    batch_shape = probs.shape[:-1]
    probs_2d = probs.view(-1, probs.size(-1))
    sampled_2d = torch.multinomial(probs_2d, num_samples=1)
    return sampled_2d.view(*batch_shape)

def generate_video(prompt, output_name="generated_video.mp4"):
    config = configparser.ConfigParser()
    config.read('configs/video_config.ini')
    v_cfg = config['VIDEO_MODEL']
    t_cfg = config['TEXT_ENCODER']
    vae_cfg = config['VAE']
    train_cfg = config['TRAINING']
    data_cfg = config['DATA']

    # Enforce textual dimensions from master config to prevent drift
    base_cfg = configparser.ConfigParser()
    base_cfg.read('configs/config.ini')
    if base_cfg.has_section('MODEL'):
        t_cfg['vocab_size'] = base_cfg['MODEL'].get('vocab_size', t_cfg.get('vocab_size', '2096'))
        t_cfg['max_seq_len'] = base_cfg['MODEL'].get('max_seq_len', t_cfg.get('max_seq_len', '64'))
    
    data_cfg['width'] = train_cfg.get('width', data_cfg.get('width', '64'))
    data_cfg['height'] = train_cfg.get('height', data_cfg.get('height', '64'))
    v_cfg['codebook_size'] = vae_cfg.get('codebook_size', v_cfg.get('codebook_size', '4096'))
    v_cfg['bos_id'] = v_cfg.get('bos_id', v_cfg['codebook_size'])
    if int(v_cfg['bos_id']) != int(v_cfg['codebook_size']):
        print(f"Aligning video BOS id to codebook size ({v_cfg['codebook_size']}) for generation.")
        v_cfg['bos_id'] = v_cfg['codebook_size']

    device = 'cpu'
    
    num_threads = int(train_cfg.get('num_threads', 0))
    safe_set_torch_threads(num_threads)
    
    vocab_cfg = configparser.ConfigParser()
    vocab_cfg.read('configs/config.ini')
    tokenizer = CharTokenizer()
    if not tokenizer.load(vocab_cfg['DATA']['vocab_path']):
        raise FileNotFoundError(f"Tokenizer not found at {vocab_cfg['DATA']['vocab_path']}")

    vae = VideoVAE(vae_cfg).to(device)
    text_encoder = TextEncoder(t_cfg).to(device)
    video_model = MultiScaleVideoModel(v_cfg).to(device)
    
    ckpt_path = train_cfg.get('checkpoint_path', 'model/video_model/video_checkpoint.pth')
    vae_loaded = False
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        video_model.load_state_dict(ckpt['video_model'])
        if 'text_encoder' in ckpt:
            text_encoder.load_state_dict(ckpt['text_encoder'])
        
        # Pull VAE physically bonded to AR Network if available
        if 'vae' in ckpt:
            result = vae.load_state_dict(ckpt['vae'], strict=False)
            allowed = {'quantizer.ema_cluster_size', 'quantizer.ema_w'}
            bad_missing = [key for key in result.missing_keys if key not in allowed]
            bad_unexpected = [key for key in result.unexpected_keys if key not in allowed]
            if bad_missing or bad_unexpected:
                raise RuntimeError(f"Bonded VAE checkpoint mismatch. Missing: {bad_missing[:8]}, unexpected: {bad_unexpected[:8]}")
            vae_loaded = True
            print(f"Loaded Video AR checkpoints (including bonded VAE) from {ckpt_path}")
        else:
            print(f"Loaded Video AR checkpoints from {ckpt_path}")
    else:
        print("Warning: No Video AR checkpoint found. Generating with random weights.")
        
    if not vae_loaded:
        vae_ckpt_path = train_cfg.get('vae_checkpoint_path', 'model/video_model/vae_checkpoint.pth')
        if os.path.exists(vae_ckpt_path):
            load_state_dict_flexible(vae, vae_ckpt_path, key='state_dict', device=device)
            print("Loaded standalone strict VAE compression checkpoint.")
        else:
            print("Warning: No VAE checkpoint found. Video decoding will fail.")

    vae.eval()
    text_encoder.eval()
    video_model.eval()

    prompt_tokens = tokenizer.encode(prompt)[:int(t_cfg.get('max_seq_len', 64))]
    if not prompt_tokens:
        prompt_tokens = [tokenizer.pad_id]
    tokens = torch.tensor([prompt_tokens], device=device)
    with torch.no_grad():
        text_emb = text_encoder(tokens)

    # Sequence dimensions based unconditionally on physics output targets
    fps = float(data_cfg.get('fps', 8.0))
    duration = float(data_cfg.get('duration', 2.0))
    total_physical_frames = int(fps * duration)
    temporal_ds = int(vae_cfg.get('temporal_downsample', 2))
    spatial_ds = int(vae_cfg.get('spatial_downsample', 8))
    width = int(train_cfg['width'])
    height = int(train_cfg['height'])
    if width % spatial_ds != 0 or height % spatial_ds != 0:
        raise ValueError(f"Video width/height must be divisible by spatial_downsample={spatial_ds}. Current: {width}x{height}.")
    if total_physical_frames <= 0 or total_physical_frames % temporal_ds != 0:
        raise ValueError(f"fps * duration ({total_physical_frames}) must be positive and divisible by temporal_downsample={temporal_ds}.")
    
    t_len = total_physical_frames // temporal_ds
    h_len = height // spatial_ds
    w_len = width // spatial_ds
    codebook_size = int(vae_cfg.get('codebook_size', 4096))
    
    seq_len = t_len * h_len * w_len
    if t_len <= 0 or h_len < 2 or w_len < 2:
        raise ValueError(f"Invalid latent video grid: T={t_len}, H={h_len}, W={w_len}. Check fps/duration/width/height/downsample config.")
    if max(t_len, h_len, w_len) >= int(v_cfg.get('max_seq_len', 2048)):
        raise ValueError("VIDEO_MODEL max_seq_len must be larger than the largest latent coordinate.")
    
    def generate_single_scale(scale_id, text_context, prefix=None):
        print(f"Generating Scale {scale_id}...")
        
        # Scale 0 downsamples spatial resolution by half to create a low-latency structural backbone
        scale_h = h_len // 2 if scale_id == 0 else h_len
        scale_w = w_len // 2 if scale_id == 0 else w_len
        seq_len_calc = t_len * scale_h * scale_w
        
        # Determine sequence coordinates completely
        # Row-major flattening guarantees this order:
        all_t = torch.arange(t_len, device=device).repeat_interleave(scale_h*scale_w).unsqueeze(0)
        all_h = torch.arange(scale_h, device=device).repeat(t_len).repeat_interleave(scale_w).unsqueeze(0)
        all_w = torch.arange(scale_w, device=device).repeat(t_len*scale_h).unsqueeze(0)
        
        frame_size = scale_h * scale_w
        indices = torch.full((1, frame_size), video_model.bos_id, device=device, dtype=torch.long)
        
        # Context building
        if prefix is not None:
            # Scale 1 dynamically pools the timeline out of the Scale 0 structural sequence
            # This crushes cross-attention RAM bloat dramatically during long scale cascade execution!
            hc = h_len // 2
            wc = w_len // 2
            coarse_emb = video_model.tok_embed(prefix.view(1, -1))
            coarse_emb = coarse_emb.view(1, t_len, hc * wc, -1).mean(dim=1)
            context_emb = torch.cat([text_context, coarse_emb], dim=1)
        else:
            context_emb = text_context
            
        kv_caches = None
        
        with torch.no_grad():
            # Generate entire frames progressively mathematically eliminating O(H*W) sequence bottlenecks
            for i in range(t_len):
                
                # We always map the absolute latest complete Frame representation!
                curr_frame_idx = indices[:, -frame_size:]
                
                t_c = all_t[:, i*frame_size:(i+1)*frame_size]
                h_c = all_h[:, i*frame_size:(i+1)*frame_size]
                w_c = all_w[:, i*frame_size:(i+1)*frame_size]
                
                scale_tensor = torch.full((1,), scale_id, device=device, dtype=torch.long)
                
                # Extremely fast O(T) generation framework passing entire structures simultaneously
                logits, kv_caches = video_model(curr_frame_idx, context_emb, scale_tensor, (t_len, scale_h, scale_w), 
                                                t_c, h_c, w_c, use_cache=True, kv_caches=kv_caches)
                
                # Logits natively represent the target frame array dimension logic
                next_frame_idx = sample_top_k_top_p(logits)
                indices = torch.cat([indices, next_frame_idx], dim=1)
                
                print(f"  Frame Cascade Phase: {i+1}/{t_len}")
                    
        return indices[:, frame_size:] # Drop BOS Frame

    # Multi-Scale Hierarchical generation Loop
    # 1. Generate coarse semantic structure
    coarse_indices = generate_single_scale(0, text_emb)
    
    # 2. Refine (Scale 1) -> Feed the coarse structure in as Cross-Attention!
    fine_indices = generate_single_scale(1, text_emb, prefix=coarse_indices)

    # Decode Latents
    fine_indices = fine_indices.view(1, t_len, h_len, w_len)
    with torch.no_grad():
        # Get embeddings from codebook
        quantized = vae.quantizer.embedding(fine_indices) # (B, T, H, W, D)
        quantized = quantized.permute(0, 4, 1, 2, 3) # (B, D, T, H, W)
        video = vae.decode(quantized) # (B, 3, T, H, W)
        
    video = video[0].permute(1, 2, 3, 0).cpu() # (T, H, W, 3)
    video = ((video + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    
    os.makedirs(data_cfg['output_path'], exist_ok=True)
    out_path = os.path.join(data_cfg['output_path'], output_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_t, video_h, video_w, _ = video.shape
    fps = float(data_cfg.get('fps', 8.0))
    out = cv2.VideoWriter(out_path, fourcc, fps, (video_w, video_h))
    
    for frame in video:
        # Pytorch handles RGB, but cv2's C++ native backend strictly requires BGR matrices
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print(f"Video natively accelerated to {out_path}")

if __name__ == "__main__":
    p = input("Enter prompt: ")
    generate_video(p)
