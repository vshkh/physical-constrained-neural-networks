import torch
import torch.nn as nn 

class Noise(nn.Module):
    def __init__(self, mode: str = "off", sigma_add: float = 0.0, sigma_mul: float = 0.0, complex_mode: bool = False):
        super().__init__()

        assert mode in ("off", "add", "mul", "both")
   
        self.mode = mode
        self.sigma_add = sigma_add
        self.sigma_mul = sigma_mul
        #self.complex_mode = complex_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off" or not self.training or x.is_complex():
            return x

        if self.mode in ("add", "both"):
            noise = torch.randn_like(x) * self.sigma_add
            x += noise
        if self.mode in ("mul", "both"):
            noise = torch.randn_like(x) * self.sigma_mul
            x *= (1 + noise)
        return x