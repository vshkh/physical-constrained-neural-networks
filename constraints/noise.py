# noise.py

import torch
import torch.nn as nn 

class Noise(nn.Module):
    def __init__(self, 
                 mode: str = "off", 
                 sigma_add: float = 0.0, 
                 sigma_mul: float = 0.0, 
                 apply_in_eval: bool = False,
                 complex_mode:  bool = False,
                 sigma_phase: float = 0.0):
        super().__init__()

        assert mode in ("off", "add", "mul", "both")
   
        self.mode = mode
        self.sigma_add = sigma_add
        self.sigma_mul = sigma_mul
        self.apply_in_eval = apply_in_eval
        self.complex_mode = complex_mode
        self.sigma_phase = sigma_phase

    @staticmethod # static method, generates complex Gaussian noise with sigma^2 var.
    def _complex_normal_like(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            # zero tensor
            return torch.zeros_like(x)
        
        # real and imag, times std dev sigma/sqrt(2)
        r = torch.randn_like(x.real) * (sigma / 2**0.5)
        i = torch.randn_like(x.real) * (sigma / 2**0.5)

        # complex tensor
        return torch.complex(r, i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        # Allow in eval only if explicitly requested
        if (not self.training) and (not self.apply_in_eval):
            return x

        # Heartbeat (rare, safe for both real/complex)
        if self.training and (torch.rand(()) < 0.001):
            try:
                m = (x.abs().mean() if x.is_complex() else x.mean()).item()
                s = (x.abs().std()  if x.is_complex() else x.std()).item()
                print(f"[NOISE ACTIVE] mode={self.mode}, add={self.sigma_add}, mul={self.sigma_mul}, mean={m:.3f}, std={s:.3f}")
            except Exception:
                pass

        # === Complex path (field-level) ===
        if x.is_complex():
            if not self.complex_mode:
                return x  # do nothing in complex unless enabled

            # Optional pure phase jitter (unit magnitude complex factor)
            if self.sigma_phase > 0.0:
                phi = torch.randn_like(x.real) * self.sigma_phase  # radians
                phase = torch.polar(torch.ones_like(phi), phi)     # cos+ i sin
                x = x * phase

            # Additive complex Gaussian (amplitude+phase)
            if self.mode in ("add", "both") and self.sigma_add > 0.0:
                x = x + Noise._complex_normal_like(x, self.sigma_add)

            # Multiplicative real gain noise (amplitude)
            if self.mode in ("mul", "both") and self.sigma_mul > 0.0:
                gain = 1.0 + torch.randn_like(x.real) * self.sigma_mul
                x = x * gain
            return x

        # === Real path (electronics-style) ===
        if self.mode in ("add", "both") and self.sigma_add > 0.0:
            x = x + torch.randn_like(x) * self.sigma_add
        if self.mode in ("mul", "both") and self.sigma_mul > 0.0:
            x = x * (1.0 + torch.randn_like(x) * self.sigma_mul)
        return x
