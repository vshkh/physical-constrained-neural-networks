# model.py

from typing import Optional
import torch
import torch.nn as nn
from constraints.noise import Noise
from constraints.quantization import ADC

# Define a complex activation function:
# - Alternatives include: split activation, a real and complex ReLu (2 performed)
# - Modulus-based activation, preserving phase and modifying amplitude.
# - Physics: Saturable absorber model or Kerr linearities can inspire complex systems; TPA

def complex_activation(z: torch.Tensor, eps : float = 1e-6) -> torch.Tensor:
    amp = torch.abs(z)
    scale = torch.tanh(amp) / (amp + eps)
    return z * scale

# Defining a linear layer capable of both real and complex parameters:

class LinearRC(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_complex: bool, bias: bool = True):
        super().__init__()

        # Use this as a flag to determine if the layers are complex or not.
        self.use_complex = use_complex
        # model.py -> LinearRC.__init__
        dtype = torch.complex64 if use_complex else torch.float32
        fan_in = in_features
        if use_complex:
            # draw real/imag ~ N(0, 1/fan_in) so E|Wz|^2 stays ~ const
            Wr = torch.randn(in_features, out_features) / (fan_in ** 0.5)
            Wi = torch.randn(in_features, out_features) / (fan_in ** 0.5)
            self.W = nn.Parameter(torch.complex(Wr, Wi).to(dtype))
        else:
            W = torch.randn(in_features, out_features) / (fan_in ** 0.5)
            self.W = nn.Parameter(W.to(dtype))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_features, dtype=dtype))
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
                 mode_quant: str = "off",
                 noise_sigma_add: float = 0.0,
                 noise_sigma_mult: float = 0.0,
                 noise_apply_in_eval: bool = False,
                 noise_sigma_phase: float = 0.0,
                 act_bits: int = 8,
                 adc_in_range=(0.0, 1.0),
                 adc_out_range=(0.0, 16.0),
                 adc_apply_in_eval: bool = False,
                 use_complex: bool = False, 
                 width: int = 256,
                 in_dim: int = 784,
                 num_classes: int = 10):
        super().__init__()

        # Flag for complex:
        self.use_complex = use_complex

        # Layers:
        self.l1 = LinearRC(in_dim, width, use_complex)
        self.l2 = LinearRC(width, num_classes, use_complex)

        # Noise:
        self.noise = Noise(
        mode_noise,
        noise_sigma_add,
        noise_sigma_mult,
        apply_in_eval=noise_apply_in_eval,       
        complex_mode=self.use_complex,
        sigma_phase=noise_sigma_phase,              # keep 0.0 to start; we can sweep later
    )

        # ADCs at entry/exit
        self.adc_in = ADC(act_bits, adc_in_range[0], adc_in_range[1], adc_apply_in_eval) if mode_quant in ("act","both") else None
        self.adc_out = ADC(act_bits, adc_out_range[0], adc_out_range[1], adc_apply_in_eval) if mode_quant in ("act","both") else None

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input image:
        # - x.size(0) is the batch size
        # - -1 is inferred from other dimensions
        x = x.view(x.size(0), -1)

        if self.adc_in is not None:
            x = self.adc_in(x)

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

        if self.adc_out is not None:
            z = self.adc_out(z)

        return z

