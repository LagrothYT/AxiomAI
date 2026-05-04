"""
AxiomAI GGUF Translation Module
================================
Converts native PyTorch .pth checkpoints into the GGUF container format
using the custom 'axiom' architecture identifier.

Supports:
  - Text Model (Base Pre-Train / SFT Fine-Tune)
  - Video VAE (3D VQ-VAE)
  - Video Text Encoder
  - Video AR Model (Spatio-Temporal Transformer)
"""

import os
import sys
import json
import configparser
import numpy as np
import torch

# Force UTF-8 output on Windows to prevent emoji crashes on cp1252 consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("❌ FATAL: 'gguf' library not installed. Run: pip install gguf")
    sys.exit(1)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def format_param_count(n):
    if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000: return f"{n / 1_000:.1f}K"
    return str(n)


def safe_numpy(tensor):
    """Convert a PyTorch tensor to a float32 numpy array, handling GPU and grad edge cases."""
    return tensor.detach().cpu().numpy().astype(np.float32)


# =============================================================================
# Tokenizer Extraction
# =============================================================================

def extract_tokenizer_metadata(vocab_path):
    """
    Reads the CharTokenizer vocab.json and returns arrays suitable for
    GGUF tokenizer embedding:
      - tokens: list of byte-string token representations
      - scores: list of floats (merge rank priority, lower = earlier merge)
      - token_types: list of ints (0=normal, 3=control for special tokens)
      - special IDs: bos_id, eos_id, pad_id
    """
    if not os.path.exists(vocab_path):
        return None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    merges = {}
    for pair_str, new_id in data['merges'].items():
        a, b = map(int, pair_str.split(','))
        merges[(a, b)] = new_id

    pad_id = data.get('pad_id')
    bos_id = data.get('bos_id')
    eos_id = data.get('eos_id')

    # Build ordered token list indexed by ID
    max_id = max(vocab.keys())
    tokens = []
    scores = []
    token_types = []

    special_ids = {pad_id, bos_id, eos_id}

    for i in range(max_id + 1):
        if i in vocab:
            tokens.append(vocab[i])

            # Score: byte-level tokens get 0.0, merged tokens get their merge rank
            # Lower rank = created earlier = higher priority
            if i < 256:
                scores.append(0.0)
            elif i in special_ids:
                scores.append(-1000.0)  # special tokens sort last
            else:
                # Find the merge rank from the merge dict values
                # The merge that created token i has value i
                scores.append(float(i - 256))

            # Token type: 0=normal, 3=control
            if i in special_ids:
                token_types.append(3)  # control token
            else:
                token_types.append(0)  # normal token
        else:
            tokens.append(bytes(f"<UNUSED_{i}>", 'utf-8'))
            scores.append(-9999.0)
            token_types.append(0)

    # Build merge strings in order: "token_a token_b"
    merge_list = sorted(merges.items(), key=lambda x: x[1])
    merge_strs = []
    for (a, b), _ in merge_list:
        tok_a = vocab.get(a, b'?')
        tok_b = vocab.get(b, b'?')
        # Serialize raw bytes separated by ASCII space to avoid stringification corruption
        merge_strs.append(tok_a + b' ' + tok_b)

    return {
        'tokens': tokens,
        'scores': scores,
        'token_types': token_types,
        'merges': merge_strs,
        'bos_id': bos_id,
        'eos_id': eos_id,
        'pad_id': pad_id,
    }


def write_tokenizer(writer, tok_data):
    """Embeds the full tokenizer into a GGUFWriter."""
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tok_data['tokens'])
    writer.add_token_scores(tok_data['scores'])
    writer.add_token_types(tok_data['token_types'])
    writer.add_token_merges(tok_data['merges'])

    if tok_data['bos_id'] is not None:
        writer.add_bos_token_id(tok_data['bos_id'])
    if tok_data['eos_id'] is not None:
        writer.add_eos_token_id(tok_data['eos_id'])
    if tok_data['pad_id'] is not None:
        writer.add_pad_token_id(tok_data['pad_id'])


# =============================================================================
# Text Model Export (Base + SFT)
# =============================================================================

def export_text_model(is_sft=False):
    """
    Exports the AxiomAI Text Transformer to GGUF.
    
    Checkpoint layout (from trainer.py):
        {
            'model': state_dict,       # <-- the weights
            'optimizer': ...,
            'scheduler': ...,
            'epoch': int,
            'best_val_loss': float,
            'config': dict             # <-- embedded config snapshot
        }
    
    state_dict keys for a 3-layer dense model:
        tok_emb.weight                     (vocab_size, d_model)
        blocks.{i}.norm1.weight            (d_model,)
        blocks.{i}.attn.wq.weight          (d_model, d_model)
        blocks.{i}.attn.wk.weight          (d_model, d_model)
        blocks.{i}.attn.wv.weight          (d_model, d_model)
        blocks.{i}.attn.wo.weight          (d_model, d_model)
        blocks.{i}.norm2.weight            (d_model,)
        blocks.{i}.ffn.w1.weight           (hidden, d_model)   SwiGLU gate
        blocks.{i}.ffn.w2.weight           (d_model, hidden)   SwiGLU down
        blocks.{i}.ffn.w3.weight           (hidden, d_model)   SwiGLU up
        norm_f.weight                      (d_model,)
        output.weight                      (vocab_size, d_model) [TIED to tok_emb]
    
    MoE variant replaces ffn.w{1,2,3} with:
        blocks.{i}.ffn.router.weight                    (num_experts, d_model)
        blocks.{i}.ffn.experts.{j}.w1.weight            (hidden, d_model)
        blocks.{i}.ffn.experts.{j}.w2.weight            (d_model, hidden)
        blocks.{i}.ffn.experts.{j}.w3.weight            (hidden, d_model)
    """
    label = "SFT" if is_sft else "Base"

    config = configparser.ConfigParser()
    config.read('configs/config.ini')

    if is_sft:
        sft_cfg = configparser.ConfigParser()
        sft_cfg.read('configs/sft_config.ini')
        checkpoint_path = sft_cfg['TRAINING'].get('checkpoint_path', 'model/sft_best_model.pth')
    else:
        checkpoint_path = config['TRAINING'].get('checkpoint_path', 'model/best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"  ❌ {label} checkpoint not found at: {checkpoint_path}")
        return False

    print(f"  Loading {label} checkpoint from {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return False
    state_dict = ckpt['model']

    # FIX #2: Prefer checkpoint's embedded config snapshot over live config file
    # This prevents config drift if the user changed config.ini after training
    ckpt_cfg = ckpt.get('config', {})
    m_cfg = config['MODEL']
    d_model = int(ckpt_cfg.get('d_model', m_cfg['d_model']))
    n_layers = int(ckpt_cfg.get('n_layers', m_cfg['n_layers']))
    n_heads = int(ckpt_cfg.get('n_heads', m_cfg['n_heads']))
    vocab_size = int(ckpt_cfg.get('vocab_size', m_cfg['vocab_size']))
    max_seq_len = int(ckpt_cfg.get('max_seq_len', m_cfg.get('max_seq_len', '256')))
    norm_eps = float(ckpt_cfg.get('norm_eps', m_cfg.get('norm_eps', '1e-6')))
    use_moe = str(ckpt_cfg.get('use_moe', m_cfg.get('use_moe', 'False'))).lower() in ('true', '1', 'yes')
    num_experts = int(ckpt_cfg.get('num_experts', m_cfg.get('num_experts', '4')))
    experts_per_tok = int(ckpt_cfg.get('experts_per_token', m_cfg.get('experts_per_token', '2')))

    if d_model % n_heads != 0:
        print(f"  ❌ Corruption Detected: d_model ({d_model}) is not perfectly cleanly divisible by n_heads ({n_heads})")
        return False
    head_dim = d_model // n_heads

    # FIX #1: Derive hidden_dim from actual tensor shape, not config computation
    # Prevents silent metadata corruption if hidden_mult was changed post-training
    hidden_dim_key = 'blocks.0.ffn.w1.weight'
    moe_hidden_key = 'blocks.0.ffn.experts.0.w1.weight'
    if hidden_dim_key in state_dict:
        hidden_dim = state_dict[hidden_dim_key].shape[0]
    elif moe_hidden_key in state_dict:
        hidden_dim = state_dict[moe_hidden_key].shape[0]
    else:
        hidden_mult = float(ckpt_cfg.get('hidden_mult', m_cfg.get('hidden_mult', '4.0')))
        hidden_dim = int(d_model * hidden_mult)

    # FIX #7: Subtract tied output.weight from param count to report true unique params
    num_params = sum(t.numel() for t in state_dict.values())
    if 'output.weight' in state_dict:
        num_params -= state_dict['output.weight'].numel()

    tag = "sft" if is_sft else "base"
    out_dir = "model/gguf/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"axiom_text_{tag}.gguf")

    print(f"  Writing GGUF to {out_path}...")
    writer = GGUFWriter(out_path, arch="llama")

    # FIX #9: Wrap writer operations in try/finally to prevent file handle leaks
    try:
        # --- Header Metadata ---
        writer.add_name(f"AxiomAI Text Model ({label})")
        writer.add_description(f"AxiomAI {label} Transformer exported from PyTorch")
        writer.add_context_length(max_seq_len)
        writer.add_embedding_length(d_model)
        writer.add_block_count(n_layers)
        writer.add_head_count(n_heads)
        writer.add_head_count_kv(n_heads)  # no GQA — full MHA
        writer.add_feed_forward_length(hidden_dim)
        writer.add_vocab_size(vocab_size)
        writer.add_layer_norm_rms_eps(norm_eps)
        writer.add_rope_dimension_count(head_dim)
        writer.add_rope_freq_base(10000.0)
        writer.add_file_type(0)  # F32

        # Custom AxiomAI metadata
        writer.add_bool("axiom.weight_tied", True)
        writer.add_bool("axiom.use_moe", use_moe)
        if use_moe:
            writer.add_expert_count(num_experts)
            writer.add_expert_used_count(experts_per_tok)

        # Best validation loss if available
        best_val = ckpt.get('best_val_loss', None)
        if best_val is not None and best_val != float('inf'):
            writer.add_float32("axiom.best_val_loss", float(best_val))

        # --- Tokenizer ---
        vocab_path = config['DATA'].get('vocab_path', 'tokenizer/vocab.json')
        tok_data = extract_tokenizer_metadata(vocab_path)
        if tok_data:
            write_tokenizer(writer, tok_data)
            print(f"  ✅ Tokenizer embedded ({len(tok_data['tokens'])} tokens, {len(tok_data['merges'])} merges)")
        else:
            print(f"  ⚠  Tokenizer not found at {vocab_path}, skipping embedding")

        # --- Tensors ---
        for name, tensor in state_dict.items():
            # Skip RoPE buffers — these are runtime-computed, not learned weights
            if 'rope_cos' in name or 'rope_sin' in name:
                continue

            # Skip KV cache states
            if 'cache_k' in name or 'cache_v' in name:
                continue

            # Llama Tensor Translation Matrix
            llama_name = name
            
            # Base embeddings
            if name == 'tok_emb.weight':
                llama_name = 'token_embd.weight'
            elif name == 'norm_f.weight':
                llama_name = 'output_norm.weight'
            elif name == 'output.weight':
                llama_name = 'output.weight'
                
            # Block elements
            elif name.startswith('blocks.'):
                parts = name.split('.')
                layer_id = parts[1]
                sub_layer = parts[2]
                
                if sub_layer == 'norm1':
                    llama_name = f'blk.{layer_id}.attn_norm.weight'
                elif sub_layer == 'norm2':
                    llama_name = f'blk.{layer_id}.ffn_norm.weight'
                elif sub_layer == 'attn':
                    proj = parts[3]
                    if proj == 'wq': llama_name = f'blk.{layer_id}.attn_q.weight'
                    elif proj == 'wk': llama_name = f'blk.{layer_id}.attn_k.weight'
                    elif proj == 'wv': llama_name = f'blk.{layer_id}.attn_v.weight'
                    elif proj == 'wo': llama_name = f'blk.{layer_id}.attn_output.weight'
                elif sub_layer == 'ffn':
                    if len(parts) == 5:
                        proj = parts[3]
                        if proj == 'w1': llama_name = f'blk.{layer_id}.ffn_gate.weight'
                        elif proj == 'w2': llama_name = f'blk.{layer_id}.ffn_down.weight'
                        elif proj == 'w3': llama_name = f'blk.{layer_id}.ffn_up.weight'
                    else:
                        raise ValueError(f"MoE export not mapped for layer: {name}")

            writer.add_tensor(llama_name, safe_numpy(tensor))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return False
    finally:
        writer.close()

    file_size = os.path.getsize(out_path)
    print(f"  ✅ {label} Model exported successfully!")
    print(f"     {format_param_count(num_params)} params → {file_size / 1024 / 1024:.2f} MB")
    print(f"     Output: {out_path}")
    return True


# =============================================================================
# Video VAE Export
# =============================================================================

def export_video_vae():
    """
    Exports the Video VQ-VAE checkpoint to GGUF.
    
    VAE state_dict keys:
        encoder.{i}.conv.weight / .bias          CausalConv3d layers
        encoder.{i}.norm1.weight / .bias          GroupNorm
        encoder.{i}.conv1/conv2.conv.weight       ResnetBlock3d
        encoder.{i}.shortcut.weight               Skip connections
        quantizer.embedding.weight                (codebook_size, latent_channels)
        decoder.{i}.*                             Mirror of encoder
    """
    v_cfg = configparser.ConfigParser()
    if not os.path.exists('configs/video_config.ini'):
        print("  ❌ video_config.ini not found")
        return False
    v_cfg.read('configs/video_config.ini')

    # FIX #8: Guard against missing sections in malformed ini
    if not v_cfg.has_section('TRAINING'):
        print("  ❌ video_config.ini is missing [TRAINING] section")
        return False

    vae_path = v_cfg['TRAINING'].get('vae_checkpoint_path', 'model/video_model/vae_checkpoint.pth')
    if not os.path.exists(vae_path):
        print(f"  ❌ VAE checkpoint not found at: {vae_path}")
        return False

    print(f"  Loading VAE checkpoint from {vae_path}...")
    # FIX #6: Wrap load in try/except for corrupt checkpoints
    try:
        state_dict = torch.load(vae_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ❌ Failed to load VAE checkpoint: {e}")
        return False
    # VAE checkpoints may be raw state_dicts or wrapped with metadata.
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    vae_cfg = v_cfg['VAE']
    
    # FIX: Derive dimensions natively out of geometry bounds to block metadata drift
    if 'quantizer.embedding.weight' in state_dict:
        codebook_size = state_dict['quantizer.embedding.weight'].shape[0]
        latent_channels = state_dict['quantizer.embedding.weight'].shape[1]
    else:
        latent_channels = int(vae_cfg.get('latent_channels', '4'))
        codebook_size = int(vae_cfg.get('codebook_size', '4096'))
        
    spatial_ds = int(vae_cfg.get('spatial_downsample', '8'))
    temporal_ds = int(vae_cfg.get('temporal_downsample', '2'))

    num_params = sum(t.numel() for t in state_dict.values())

    out_dir = "model/gguf/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "axiom_video_vae.gguf")

    print(f"  Writing GGUF to {out_path}...")
    writer = GGUFWriter(out_path, arch="axiom_vae")

    try:
        writer.add_name("AxiomAI Video VQ-VAE")
        writer.add_description("3D Vector-Quantized VAE for video latent compression")
        writer.add_file_type(0)  # F32

        # VAE-specific metadata
        writer.add_uint32("axiom_vae.latent_channels", latent_channels)
        writer.add_uint32("axiom_vae.codebook_size", codebook_size)
        writer.add_uint32("axiom_vae.spatial_downsample", spatial_ds)
        writer.add_uint32("axiom_vae.temporal_downsample", temporal_ds)

        for name, tensor in state_dict.items():
            writer.add_tensor(name, safe_numpy(tensor))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return False
    finally:
        writer.close()

    file_size = os.path.getsize(out_path)
    print(f"  ✅ Video VAE exported successfully!")
    print(f"     {format_param_count(num_params)} params → {file_size / 1024 / 1024:.2f} MB")
    print(f"     Output: {out_path}")
    return True


# =============================================================================
# Video Text Encoder Export
# =============================================================================

def export_video_text_encoder():
    """
    Exports the standalone Video Domain Text Encoder to GGUF.

    state_dict keys:
        tok_emb.weight                   (vocab_size, d_model)
        pos_emb                          (1, max_seq_len, d_model)
        blocks.{i}.norm1.weight          (d_model,)
        blocks.{i}.attn.wq/wk/wv/wo.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.w1/w2/w3.weight
        norm.weight                      (d_model,)
        norm.bias                        (d_model,)
    """
    v_cfg = configparser.ConfigParser()
    if not os.path.exists('configs/video_config.ini'):
        print("  ❌ video_config.ini not found")
        return False
    v_cfg.read('configs/video_config.ini')

    if not v_cfg.has_section('TRAINING'):
        print("  ❌ video_config.ini is missing [TRAINING] section")
        return False

    te_path = v_cfg['TRAINING'].get('text_encoder_checkpoint_path',
                                     'model/video_model/text_encoder_checkpoint.pth')
    if not os.path.exists(te_path):
        print(f"  ❌ Text Encoder checkpoint not found at: {te_path}")
        return False

    print(f"  Loading Text Encoder checkpoint from {te_path}...")
    try:
        state_dict = torch.load(te_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ❌ Failed to load Text Encoder checkpoint: {e}")
        return False
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    t_cfg = v_cfg['TEXT_ENCODER']
    # Sync vocab/seq from master config (same as video_trainer.py)
    base_cfg = configparser.ConfigParser()
    base_cfg.read('configs/config.ini')

    # Derive d_model from actual tensor shape to prevent config drift
    if 'tok_emb.weight' in state_dict:
        d_model = state_dict['tok_emb.weight'].shape[1]
        vocab_size = state_dict['tok_emb.weight'].shape[0]
    else:
        d_model = int(t_cfg.get('d_model', '128'))
        vocab_size = int(base_cfg['MODEL'].get('vocab_size', t_cfg.get('vocab_size', '2096')))

    # Derive n_layers from actual block count
    block_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]
    n_layers = max(block_indices) + 1 if block_indices else int(t_cfg.get('n_layers', '2'))

    n_heads = int(t_cfg.get('n_heads', '4'))
    
    if 'pos_emb' in state_dict:
        max_seq_len = state_dict['pos_emb'].shape[1]
    else:
        max_seq_len = int(base_cfg['MODEL'].get('max_seq_len', t_cfg.get('max_seq_len', '64')))

    if d_model % n_heads != 0:
        print(f"  ❌ Corruption Detected: Text Encoder d_model ({d_model}) is not cleanly divisible by n_heads ({n_heads})")
        return False
        
    head_dim = d_model // n_heads

    num_params = sum(t.numel() for t in state_dict.values())

    out_dir = "model/gguf/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "axiom_video_text_encoder.gguf")

    print(f"  Writing GGUF to {out_path}...")
    writer = GGUFWriter(out_path, arch="axiom_text_encoder")

    try:
        writer.add_name("AxiomAI Video Text Encoder")
        writer.add_description("Conditioning text encoder for video generation pipeline")
        writer.add_context_length(max_seq_len)
        writer.add_embedding_length(d_model)
        writer.add_block_count(n_layers)
        writer.add_head_count(n_heads)
        writer.add_head_count_kv(n_heads)
        writer.add_vocab_size(vocab_size)
        writer.add_rope_dimension_count(head_dim)
        writer.add_file_type(0)  # F32

        # Note: text encoder uses learned absolute positional embeddings, not RoPE
        writer.add_bool("axiom_text_encoder.uses_absolute_pos_emb", True)

        # Embed tokenizer since the text encoder needs it for inference
        vocab_path = base_cfg['DATA'].get('vocab_path', 'tokenizer/vocab.json')
        tok_data = extract_tokenizer_metadata(vocab_path)
        if tok_data:
            write_tokenizer(writer, tok_data)

        for name, tensor in state_dict.items():
            if 'rope_cos' in name or 'rope_sin' in name:
                continue
            if 'cache_k' in name or 'cache_v' in name:
                continue
            writer.add_tensor(name, safe_numpy(tensor))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return False
    finally:
        writer.close()

    file_size = os.path.getsize(out_path)
    print(f"  ✅ Video Text Encoder exported successfully!")
    print(f"     {format_param_count(num_params)} params → {file_size / 1024 / 1024:.2f} MB")
    print(f"     Output: {out_path}")
    return True


# =============================================================================
# Video AR Model Export
# =============================================================================

def export_video_ar_model():
    """
    Exports the full Video AR checkpoint bundle (AR model + text encoder + VAE)
    into a single GGUF file.

    The AR checkpoint from video_trainer.py contains:
        {
            'video_model': MultiScaleVideoModel state_dict,
            'text_encoder': TextEncoder state_dict,
            'vae': VideoVAE state_dict,
            'epoch': int,
            'best_val_loss': float,
            'config': dict
        }

    AR Model state_dict keys:
        tok_embed.weight                          (bos_id+1, d_model)
        scale_embed.weight                        (max_scale_levels, d_model)
        blocks.{i}.norm1/norm2/norm3.weight       RMSNorm
        blocks.{i}.norm_cross.weight              RMSNorm
        blocks.{i}.spatial_attn.wq/wk/wv/wo.weight
        blocks.{i}.temporal_attn.wq/wk/wv/wo.weight
        blocks.{i}.cross_attn.wq/wk/wv/wo.weight
        blocks.{i}.ffn.0/2.weight                 SiLU FFN
        norm_f.weight                              (d_model,)
        head.weight                               (codebook_size, d_model)
    """
    v_cfg = configparser.ConfigParser()
    if not os.path.exists('configs/video_config.ini'):
        print("  ❌ video_config.ini not found")
        return False
    v_cfg.read('configs/video_config.ini')

    if not v_cfg.has_section('TRAINING'):
        print("  ❌ video_config.ini is missing [TRAINING] section")
        return False

    ar_path = v_cfg['TRAINING'].get('checkpoint_path', 'model/video_model/video_checkpoint.pth')
    if not os.path.exists(ar_path):
        print(f"  ❌ Video AR checkpoint not found at: {ar_path}")
        return False

    print(f"  Loading Video AR checkpoint from {ar_path}...")
    try:
        ckpt = torch.load(ar_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ❌ Failed to load Video AR checkpoint: {e}")
        return False

    vm_cfg = v_cfg['VIDEO_MODEL']
    vae_cfg = v_cfg['VAE']
    data_cfg = v_cfg['DATA']

    # Secure verification parameters against embedded archive to bypass file modifications
    ckpt_cfg = ckpt.get('config', {})
    if isinstance(ckpt_cfg, dict) and 'VIDEO_MODEL' in ckpt_cfg:
        hist_vm = dict(ckpt_cfg['VIDEO_MODEL'])
    else:
        hist_vm = vm_cfg

    # Derive structural dims from actual tensors when available
    ar_sd = ckpt.get('video_model', {})
    if 'norm_f.weight' in ar_sd:
        d_model = ar_sd['norm_f.weight'].shape[0]
    else:
        d_model = int(hist_vm.get('d_model', '128'))

    if 'head.weight' in ar_sd:
        codebook_size = ar_sd['head.weight'].shape[0]
    else:
        codebook_size = int(vae_cfg.get('codebook_size', '4096'))

    block_indices = [int(k.split('.')[1]) for k in ar_sd.keys() if k.startswith('blocks.')]
    n_layers = max(block_indices) + 1 if block_indices else int(hist_vm.get('n_layers', '4'))

    n_heads = int(hist_vm.get('n_heads', '8'))
    bos_id = int(hist_vm.get('bos_id', '4096'))
    max_scale = int(hist_vm.get('max_scale_levels', '4'))
    max_seq_len = int(hist_vm.get('max_seq_len', '2048'))
    
    if d_model % n_heads != 0:
        print(f"  ❌ Corruption Detected: AR Model d_model ({d_model}) is not cleanly divisible by n_heads ({n_heads})")
        return False
        
    head_dim = d_model // n_heads
    fps = float(data_cfg.get('fps', '8.0'))
    duration = float(data_cfg.get('duration', '2.0'))

    out_dir = "model/gguf/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "axiom_video_ar.gguf")

    print(f"  Writing GGUF to {out_path}...")
    writer = GGUFWriter(out_path, arch="axiom_video")

    try:
        writer.add_name("AxiomAI Video AR Model")
        writer.add_description("Full video generation bundle: AR Transformer + Text Encoder + VAE")
        writer.add_context_length(max_seq_len)
        writer.add_embedding_length(d_model)
        writer.add_block_count(n_layers)
        writer.add_head_count(n_heads)
        writer.add_head_count_kv(n_heads)
        writer.add_vocab_size(codebook_size)
        writer.add_rope_dimension_count(head_dim)
        writer.add_rope_freq_base(10000.0)
        writer.add_file_type(0)  # F32

        # Video-specific metadata
        writer.add_uint32("axiom_video.bos_id", bos_id)
        writer.add_uint32("axiom_video.codebook_size", codebook_size)
        writer.add_uint32("axiom_video.max_scale_levels", max_scale)
        writer.add_uint32("axiom_video.spatial_downsample", int(vae_cfg.get('spatial_downsample', '8')))
        writer.add_uint32("axiom_video.temporal_downsample", int(vae_cfg.get('temporal_downsample', '2')))
        writer.add_float32("axiom_video.fps", fps)
        writer.add_float32("axiom_video.duration", duration)

        best_val = ckpt.get('best_val_loss', None)
        if best_val is not None and best_val != float('inf'):
            writer.add_float32("axiom_video.best_val_loss", float(best_val))

        # Embed tokenizer for the text-conditioning path
        base_cfg = configparser.ConfigParser()
        base_cfg.read('configs/config.ini')
        vocab_path = base_cfg['DATA'].get('vocab_path', 'tokenizer/vocab.json')
        tok_data = extract_tokenizer_metadata(vocab_path)
        if tok_data:
            write_tokenizer(writer, tok_data)

        # --- Write all sub-model tensors with namespaced prefixes ---
        total_params = 0

        # 1. Video AR Model
        if 'video_model' in ckpt:
            for name, tensor in ckpt['video_model'].items():
                if 'rope_cos' in name or 'rope_sin' in name:
                    continue
                writer.add_tensor(f"video_model.{name}", safe_numpy(tensor))
                total_params += tensor.numel()
            print(f"  ✅ AR Model tensors written ({format_param_count(total_params)})")

        # 2. Text Encoder
        te_params = 0
        if 'text_encoder' in ckpt:
            for name, tensor in ckpt['text_encoder'].items():
                if 'rope_cos' in name or 'rope_sin' in name:
                    continue
                if 'cache_k' in name or 'cache_v' in name:
                    continue
                writer.add_tensor(f"text_encoder.{name}", safe_numpy(tensor))
                te_params += tensor.numel()
            total_params += te_params
            print(f"  ✅ Text Encoder tensors written ({format_param_count(te_params)})")

        # 3. VAE
        vae_params = 0
        if 'vae' in ckpt:
            for name, tensor in ckpt['vae'].items():
                writer.add_tensor(f"vae.{name}", safe_numpy(tensor))
                vae_params += tensor.numel()
            total_params += vae_params
            print(f"  ✅ VAE tensors written ({format_param_count(vae_params)})")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return False
    finally:
        writer.close()

    file_size = os.path.getsize(out_path)
    print(f"  ✅ Video AR Bundle exported successfully!")
    print(f"     {format_param_count(total_params)} total params → {file_size / 1024 / 1024:.2f} MB")
    print(f"     Output: {out_path}")
    return True


# =============================================================================
# Interactive Menu
# =============================================================================

def export_menu():
    clear_screen()
    print("╔══════════════════════════════════════════════╗")
    print("║       G G U F   E X P O R T   M E N U       ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Scan what checkpoints exist
    config = configparser.ConfigParser()
    config.read('configs/config.ini')

    v_cfg = configparser.ConfigParser()
    has_video = os.path.exists('configs/video_config.ini')
    if has_video:
        v_cfg.read('configs/video_config.ini')
        # FIX #8: Guard against missing sections in malformed ini
        has_video = v_cfg.has_section('TRAINING')

    base_path = config['TRAINING'].get('checkpoint_path', 'model/best_model.pth')
    base_exists = os.path.exists(base_path)

    sft_cfg = configparser.ConfigParser()
    sft_cfg.read('configs/sft_config.ini')
    sft_path = sft_cfg['TRAINING'].get('checkpoint_path', 'model/sft_best_model.pth') if sft_cfg.has_section('TRAINING') else ''
    sft_exists = os.path.exists(sft_path)

    vae_path = v_cfg['TRAINING'].get('vae_checkpoint_path', '') if has_video else ''
    vae_exists = os.path.exists(vae_path) if vae_path else False

    te_path = v_cfg['TRAINING'].get('text_encoder_checkpoint_path', '') if has_video else ''
    te_exists = os.path.exists(te_path) if te_path else False

    ar_path = v_cfg['TRAINING'].get('checkpoint_path', '') if has_video else ''
    ar_exists = os.path.exists(ar_path) if ar_path else False

    s = lambda exists: "✅" if exists else "❌"

    print(f"  [ AVAILABLE CHECKPOINTS ]")
    print(f"  ─────────────────────────────────────────")
    print(f"  {s(base_exists)} 1 │ Text Model (Base Pre-Train)")
    print(f"  {s(sft_exists)} 2 │ Text Model (SFT Fine-Tune)")
    print(f"  {s(vae_exists)} 3 │ Video VAE")
    print(f"  {s(te_exists)} 4 │ Video Text Encoder")
    print(f"  {s(ar_exists)} 5 │ Video AR Model (Full Bundle)")
    print(f"       6 │ Export ALL Available")
    print(f"       0 │ Back to Main Menu")
    print()

    choice = input("  Select (0-6): ").strip()

    results = []

    if choice == '1':
        if not base_exists:
            print(f"\n  ❌ No Base checkpoint found at {base_path}")
        else:
            print()
            results.append(export_text_model(is_sft=False))

    elif choice == '2':
        if not sft_exists:
            print(f"\n  ❌ No SFT checkpoint found at {sft_path}")
        else:
            print()
            results.append(export_text_model(is_sft=True))

    elif choice == '3':
        if not vae_exists:
            print(f"\n  ❌ No VAE checkpoint found at {vae_path}")
        else:
            print()
            results.append(export_video_vae())

    elif choice == '4':
        if not te_exists:
            print(f"\n  ❌ No Text Encoder checkpoint found at {te_path}")
        else:
            print()
            results.append(export_video_text_encoder())

    elif choice == '5':
        if not ar_exists:
            print(f"\n  ❌ No Video AR checkpoint found at {ar_path}")
        else:
            print()
            results.append(export_video_ar_model())

    elif choice == '6':
        print("\n  Exporting all available checkpoints...\n")
        if base_exists:
            results.append(export_text_model(is_sft=False))
            print()
        if sft_exists:
            results.append(export_text_model(is_sft=True))
            print()
        if vae_exists:
            results.append(export_video_vae())
            print()
        if te_exists:
            results.append(export_video_text_encoder())
            print()
        if ar_exists:
            results.append(export_video_ar_model())
            print()

        if not any([base_exists, sft_exists, vae_exists, te_exists, ar_exists]):
            print("  ❌ No checkpoints found to export.")
        else:
            ok = sum(1 for r in results if r)
            total = len(results)
            print(f"\n  ── Export Complete: {ok}/{total} successful ──")

    elif choice == '0':
        return

    else:
        print("\n  Invalid selection.")


if __name__ == "__main__":
    export_menu()
