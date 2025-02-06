import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class Diffuser(nn.Module):
    def __init__(self, num_timesteps = 1000, beta_start = 0.0001, beta_end = 0.02, device = 'cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device = device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)
    
    # Add Noise -> Sampling x_t from x0 (the given t will have a shape of (N,))
    def add_noise(self, x0, t):
        T = self.num_timesteps
        # check all the sampled timestep is between 1~T(1000)
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1 

        alpha_bars = self.alpha_bars[t_idx] # (N,)
        N = alpha_bars.size(0) # Batch size
        alpha_bars = alpha_bars.view(N, 1, 1, 1)

        # Noise for reparameterization
        noise = torch.randn_like(x0, device = self.device) # (N,C,H,W)
        x_t = torch.sqrt(alpha_bars) * x0 + torch.sqrt(1 - alpha_bars) * noise

        return x_t, noise # Returns sampled x_t(noise added) and noise that has been added from x0
    
    # Denoising -> sampling x_t-1 from a given input x_t and t
    def denoise(self, model, x_t, t, labels, gamma): # x_t (N,C,H,W), t (N,)
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alphas = self.alphas[t_idx]
        alpha_bars = self.alpha_bars[t_idx]
        alpha_bars_prev = self.alpha_bars[t_idx - 1]

        N = alphas.size(0)
        alphas = alphas.view(N, 1, 1, 1)
        alpha_bars = alpha_bars.view(N, 1, 1, 1)
        alpha_bars_prev = alpha_bars_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            noise_cond = model(x_t, t, labels) # Condition score (label)
            noise_uncond = model(x_t, t) # Uncondition score
            noise = noise_uncond + gamma * (noise_cond - noise_uncond)
        model.train()

        eps = torch.randn_like(x_t, device = self.device) # (N,C,H,W)
        # All noise for reparameterize in t = 1 -> t =0 will be 0
        eps[t == 1] = 0

        mu = (x_t - noise * ((1 - alphas) / torch.sqrt(1 - alpha_bars))) / torch.sqrt(alphas) 
        std = torch.sqrt(((1 - alphas) * (1 - alpha_bars_prev)) / (1 - alpha_bars))

        return mu + std * eps
    
    # Reverse to original image from tensor
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()

        return to_pil(x)

    # Sample datas from (N,C,H,W) shape gaussian noise N(0,I)
    def sample(self, model, labels = None, x_shape = (20, 1, 28, 28), gamma = 3.0):
        T = self.num_timesteps

        # Random Gaussian Noise
        x_t = torch.randn(x_shape, device = self.device) # (20, 1, 28, 28)

        # Labels
        if labels is None: # If there's no condition, the diffusion model samples random data from the learned distribution
            labels = torch.randint(0, 10, (len(x_t),), device = self.device) # (N,) -> 0 ~ 9 classes

        # Sampling from T to 1
        for i in tqdm(range(T, 0, -1)):
            ts = torch.tensor([i] * len(x_t), device = self.device, dtype = torch.long) #(N,)
            x_t = self.denoise(model, x_t, ts, labels, gamma) # Add labels (Condition)
        
        imgs = [self.reverse_to_img(img) for img in x_t]

        return imgs, labels