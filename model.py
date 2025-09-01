# model.py

from typing import Optional
import torch
import torch.nn as nn
from constraints.noise import Noise

# Define a complex activation function:
# - Alternatives include: split activation, a real and complex ReLu (2 performed)
# - Modulus-based activation, preserving phase and modifying amplitude.
# - Physics: Saturable absorber model or Kerr linearities can inspire complex systems; TPA

def complex_activation(z: torch.Tensor) -> torch.Tensor:
    amp = torch.abs(z)
    return z * torch.tanh(amp)

# Defining a linear layer capable of both real and complex parameters:

class LinearRC(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_complex: bool, bias: bool = True):
        super().__init__()

        # Use this as a flag to determine if the layers are complex or not.
        self.use_complex = use_complex
        dtype = torch.complex64 if use_complex else torch.float32
        
        self.W = nn.Parameter(torch.randn(in_features, out_features, dtype=dtype))
        if bias:
            self.b = nn.Parameter(torch.randn(out_features, dtype=dtype))
        else:
            self.register_parameter('b', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # @ operator does matmul using NumPy
        out = x @ self.W
        if self.b is not None:
            out += self.b 
        return out

class TinyNet(nn.Module):
    def __init__(self,
                 mode_noise: str = "off", 
                 noise_sigma_add: float = 0.0,
                 noise_sigma_mult: float = 0.0,
                 use_complex: bool = False, 
                 width: int = 256):
        super().__init__()

        # Flag for complex:
        self.use_complex = use_complex
        in_dim, out_dim = 784, 10 # 784 = 28^2, 10 (0-9 classes)

        # Layers:
        self.l1 = LinearRC(in_dim, width, use_complex)
        self.l2 = LinearRC(width, out_dim, use_complex)

        # Noise:
        self.noise = Noise(mode_noise, noise_sigma_add, noise_sigma_mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input image:
        # - x.size(0) is the batch size
        # - -1 is inferred from other dimensions
        x = x.view(x.size(0), -1)

        # Convert to complex if set to use_complex
        if self.use_complex and x.dtype != torch.complex64:
            x = x.to(torch.complex64)
        
        # First transform
        z = self.l1(x)

        # Apply noise:
        z = self.noise(z)
        
        # Activation, either complex or real
        z = complex_activation(z) if self.use_complex else torch.relu(z)

        # Second transform
        z = self.l2(z)

        # Convert to real logits if complex via abs
        if self.use_complex:
            z = torch.abs(z)**2

        return z

