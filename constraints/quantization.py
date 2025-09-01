# quantization.py

import torch
import torch.nn as nn 

# Goal: map a continuous signal to one of 2^n discrete codes:

# Rounding isn't differentiable, so we need to implement a Straight-Through Estimator
# where the forward pass does quantization, and the backward pass pretends the quantizer
# is an identity function.

class UniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xmin, xmax, num_bits):
        x = torch.clamp(x, xmin, xmax)
        levels = (1 << int(num_bits)) - 1 # 2^n - 1
        scale = levels / (xmax - xmin + 1e-12)
        xq = torch.round((x - xmin)* scale) / scale + xmin
        return xq
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
    
class ADC(nn.Module):
    def __init__(self, num_bits: int, xmin: float, xmax: float, apply_in_eval: bool = False):
        super().__init__()
        assert num_bits >= 2
        self.num_bits = int(num_bits)
        self.register_buffer('xmin', torch.tensor(xmin))
        self.register_buffer('xmax', torch.tensor(xmax))
        self.apply_in_eval = bool(apply_in_eval)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.apply_in_eval:
            return UniformQuantizeSTE.apply(x, self.xmin, self.xmax, self.num_bits)
        return x