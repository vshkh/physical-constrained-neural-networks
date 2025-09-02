# drift.py

import torch
import torch.nn as nn
from typing import Optional
from model import LinearRC

"""
- Defining a process that isn't a layer in the model; an external process to mutate weights.
- It has no parameters and lives outside the module, so a plain class is more than fine.
- Noise and ADC belong inside the model on the other hand, hence why they are subclass nn.Module

- Drift mutates parameters across epochs/baches, representing how detectors change calibrated settings
- as it heats up; hence, walking away from their optimized values unless recalibrated. 
"""
class DriftController:
    # Minimal drift: per call, nudge LinearRC weights/biases by small Gaussian noise

    def __init__(self, eta: float = 0.0, mode: str = "off"):
        assert mode in ("off", "epoch", "batch")
        self.eta = eta
        self.mode = mode
    
    @torch.no_grad()
    def attach(self, model: nn.Module):
        pass 

    @torch.no_grad()
    def step_epoch(self, model: nn.Module):
        if self.mode != "epoch" or self.eta == 0.0:
            return
        self._apply_gaussian_nudge(model)

    @torch.no_grad()
    def step_batch(self, model: nn.Module):
        if self.mode != "batch" or self.eta == 0.0:
            return
        self._apply_gaussian_nudge(model)

    @torch.no_grad()
    def _apply_gaussian_nudge(self, model: nn.Module):
        # For each LinearRC, do: W <- W + eta * N(0,1), b likewise if present.
        for m in model.modules():
            if isinstance(m, LinearRC):
                if m.W is not None:
                    m.W.add_(torch.randn_like(m.W) * self.eta)
                if m.b is not None:
                    m.b.add_(torch.randn_like(m.b) * self.eta)