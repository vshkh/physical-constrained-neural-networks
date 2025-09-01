import torch
import torch.nn as nn 

class Noise(nn.Module):
    def __init__(self, mode: str = "off", sigma_add: float = 0.0, sigma_mul: float = 0.0, complex_mode: bool = False):
        super().__init__()

        assert mode in ("off", "add", "mul", "both")
   
        self.mode = mode
        self.sigma_add = sigma_add
        self.sigma_mul = sigma_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off" or not self.training or x.is_complex():
            return x
        
        # inside Noise.forward
        if self.training:
            if torch.rand(()) < 0.001:
                print(f"[NOISE ACTIVE] mode={self.mode}, add={self.sigma_add}, mul={self.sigma_mul}, x.mean={x.mean().item():.3f}, x.std={x.std().item():.3f}")

        if self.mode in ("add", "both"):
            noise = torch.randn_like(x) * self.sigma_add
            x += noise
        if self.mode in ("mul", "both"):
            noise = torch.randn_like(x) * self.sigma_mul
            x *= (1 + noise)
        return x