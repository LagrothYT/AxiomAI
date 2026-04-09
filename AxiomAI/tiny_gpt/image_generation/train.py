import os
import sys
import argparse

# Add parent directory to sys.path so we can run this script directly and resolve 'image_generation' package imports safely
# We also use this to hook into the root tinyGPT's tokenizer logic.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_generation.config import ImageModelConfig
from image_generation.scheduler import DDPMScheduler
from image_generation.model.vae import VAE
from image_generation.model.text_encoder import TextEncoder
from image_generation.model.diffusion import LatentDiffusion
from image_generation.image_loader import ImageDataset

def train_vae(config):
    print("Initiating VAE 'The Eyes' pre-training pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("USE_ROCM") == "1" and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    vae = VAE(latent_channels=config.vae_channels).to(device)
    
    save_path = os.path.join(config.output_dir_model, "vae_best.pt")
    start_epoch = 0
    if os.path.exists(save_path):
        print(f"Resuming VAE from {save_path}...")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        
    optimizer = optim.AdamW(vae.parameters(), lr=config.learning_rate)
    
    print(f"Loading dataset from {config.data_dir}...")
    dataset = ImageDataset(config.data_dir, config)
    if len(dataset) == 0:
        print(f"[!] No images found in {config.data_dir}. Add some to start training.")
        return
        
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    import torch.nn.functional as F
    best_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, config.epochs):
            vae.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for img_b, _ in loop:
                img_b = img_b.to(device)
                optimizer.zero_grad()
                reconstruction, mu, logvar = vae(img_b)
                
                recon_loss = F.mse_loss(reconstruction, img_b)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / (img_b.size(0) * config.vae_channels * config.latent_height * config.latent_width)
                loss = recon_loss + config.kl_weight * kl_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                
                loop.set_postfix(recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.4f}")
                                 
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} (Recon: {total_recon/len(train_loader):.4f}, KL: {total_kl/len(train_loader):.4f})")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({"model": vae.state_dict(), "epoch": epoch, "loss": best_loss}, save_path)
                
    except KeyboardInterrupt:
        print("\n[!] Training paused by user.")
        torch.save({"model": vae.state_dict(), "epoch": epoch if 'epoch' in locals() else start_epoch, "loss": avg_loss if 'avg_loss' in locals() else best_loss}, save_path.replace(".pt", "_interrupted.pt"))


def train_diffusion(config):
    print("Initiating Latent Diffusion 'The Brain' & Text Encoder 'The Ears' joint training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("USE_ROCM") == "1" and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    import config as text_config
    from tokenizer.bpe import BPETokenizer
    tok_path = os.path.join(text_config.tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"[!] Tokenizer missing at {tok_path}. Cannot train text embeddings.")
        return
    tokenizer = BPETokenizer.load(tok_path)
    config.vocab_size = len(tokenizer.vocab) # Force sync parameter to prevent Embedding crash
    
    vae = VAE(latent_channels=config.vae_channels).to(device)
    vae_path = os.path.join(config.output_dir_model, "vae_best.pt")
    if not os.path.exists(vae_path):
        print(f"[!] Frozen VAE missing at {vae_path}. You must finish Option 6 first.")
        return
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=False)["model"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    text_encoder = TextEncoder(config.vocab_size, config.text_embedding_dim, config.text_max_seq_len).to(device)
    diffusion = LatentDiffusion(in_channels=config.diffusion_channels, text_embed_dim=config.text_embedding_dim).to(device)
    scheduler = DDPMScheduler()
    
    params = list(text_encoder.parameters()) + list(diffusion.parameters())
    optimizer = optim.AdamW(params, lr=config.learning_rate)
    
    save_path = os.path.join(config.output_dir_model, "diffusion_best.pt")
    start_epoch = 0
    if os.path.exists(save_path):
        chk = torch.load(save_path, map_location=device, weights_only=False)
        text_encoder.load_state_dict(chk["text_encoder"])
        diffusion.load_state_dict(chk["diffusion"])
        start_epoch = chk.get("epoch", 0) + 1
        print(f"Resumed from {save_path}")
        
    dataset = ImageDataset(config.data_dir, config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    import torch.nn.functional as F
    best_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, config.epochs):
            text_encoder.train()
            diffusion.train()
            total_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for img_b, captions in loop:
                img_b = img_b.to(device)
                b_size = img_b.size(0)
                
                input_ids_list = []
                for cap in captions:
                    tokens = tokenizer.encode(cap)
                    if len(tokens) > config.text_max_seq_len:
                        tokens = tokens[:config.text_max_seq_len]
                    else:
                        tokens = tokens + [0] * (config.text_max_seq_len - len(tokens))
                    input_ids_list.append(tokens)
                input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                
                # 1. Joint encode text
                context = text_encoder(input_ids)
                
                # 2. Frozen Stage 1 execution
                with torch.no_grad():
                    latents, _, _ = vae.encode(img_b)
                
                # 3. Add cosine noise map schedule
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (b_size,), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # 4. Predict
                pred_noise = diffusion(noisy_latents, timesteps, context)
                loss = F.mse_loss(pred_noise, noise)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({"text_encoder": text_encoder.state_dict(), "diffusion": diffusion.state_dict(), "epoch": epoch, "loss": best_loss}, save_path)
                
    except KeyboardInterrupt:
        print("\n[!] Training paused by user.")


def _dry_run(config):
    print("Running initial sanity check on architectures...")
    vae = VAE(latent_channels=config.vae_channels)
    
    # We must mock encoder encode -> decode output lengths for safety
    dummy_imgs = torch.randn(2, 3, config.image_height, config.image_width)
    latents, mu, logvar = vae.encode(dummy_imgs)
    rec = vae.decode(latents)
    assert rec.shape == dummy_imgs.shape, "Decoder output shape mismatch."
    assert latents.shape == (2, config.vae_channels, config.latent_height, config.latent_width), "Latent math mismatch."
    print("Shape checks passed perfectly. VAE is functional.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["vae", "diffusion", "dry-run"], default="dry-run")
    args = parser.parse_args()
    
    config = ImageModelConfig.load()
    
    if args.mode == "vae":
        train_vae(config)
    elif args.mode == "diffusion":
        train_diffusion(config)
    elif args.mode == "dry-run":
        _dry_run(config)
