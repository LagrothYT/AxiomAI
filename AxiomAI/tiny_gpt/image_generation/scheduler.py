import torch
import math

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        self.betas = self._cosine_beta_schedule(num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(0, num_train_timesteps).flip(0)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        # Base DDPM single step (must iterate all 1000)
        device = model_output.device
        t = timestep
        alpha_t = self.alphas[t].to(device)
        alpha_cumprod_t = self.alphas_cumprod[t].to(device)
        beta_t = self.betas[t].to(device)
        
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod_t) ** 0.5
        pred_mean = (1 / (alpha_t ** 0.5)) * (sample - (beta_t / sqrt_one_minus_alpha_prod) * model_output)
        
        if t > 0:
            noise = torch.randn_like(model_output)
            sigma = beta_t ** 0.5 
            return pred_mean + sigma * noise
        return pred_mean

    def step_ddim(self, model_output, timestep, prev_timestep, sample, eta=0.0):
        # DDIM solver allows jumping steps without breaking the variance math
        device = model_output.device
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(device) if prev_timestep >= 0 else torch.tensor(1.0).to(device)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 1. Compute predicted x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # 2. Compute variance 
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # 3. Compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).clamp(min=0) ** 0.5 * model_output
        
        # 4. Compute x_{t-1} or x_{t-n} smoothly
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample = prev_sample + std_dev_t * noise
            
        return prev_sample
