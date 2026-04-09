import os
import json
import dataclasses
from dataclasses import dataclass

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ImageModelConfig:
    # Image dimensions (width by height strictly requested)
    image_width: int = 144
    image_height: int = 96
    
    # Text encoder config
    vocab_size: int = 50257
    text_embedding_dim: int = 256
    text_max_seq_len: int = 77
    
    # VAE / Latent config
    vae_channels: int = 4
    vae_downsample_factor: int = 8
    
    # Diffusion config
    diffusion_channels: int = 4 # Should match vae_channels usually
    diffusion_layers: int = 4
    num_timesteps: int = 1000
    
    # Training paths
    data_dir: str = "Photos"
    output_dir_model: str = "model"
    output_dir_image: str = "out_image"
    
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 500
    
    # VAE Loss Weight
    kl_weight: float = 1e-5

    @classmethod
    def load(cls):
        config_path = os.path.join(get_root_dir(), "image_config.json")
        instance = cls()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
            except Exception as e:
                print(f"[!] Could not load image_config.json: {e}. Using strict defaults.")
        else:
            instance.save()
            
        instance._validate()
        return instance

    def save(self):
        config_path = os.path.join(get_root_dir(), "image_config.json")
        try:
            with open(config_path, "w") as f:
                data = dataclasses.asdict(self)
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[!] Could not save image_config.json: {e}")

    def _validate(self):
        # Enforce mathematical correct dimensions for VAE downsampling
        if self.image_width % self.vae_downsample_factor != 0:
            raise ValueError(f"image_width ({self.image_width}) must be fully divisible by vae_downsample_factor ({self.vae_downsample_factor})")
        if self.image_height % self.vae_downsample_factor != 0:
            raise ValueError(f"image_height ({self.image_height}) must be fully divisible by vae_downsample_factor ({self.vae_downsample_factor})")
            
        real_model_out = os.path.join(get_root_dir(), self.output_dir_model)
        if not os.path.exists(real_model_out):
            os.makedirs(real_model_out, exist_ok=True)
            
        real_image_out = os.path.join(get_root_dir(), self.output_dir_image)
        if not os.path.exists(real_image_out):
            os.makedirs(real_image_out, exist_ok=True)
            
    @property
    def latent_width(self) -> int:
        return self.image_width // self.vae_downsample_factor
        
    @property
    def latent_height(self) -> int:
        return self.image_height // self.vae_downsample_factor
