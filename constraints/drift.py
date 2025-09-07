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
    def __init__(self, eta=0.0, mode="off", drift_bias=True, multiplicative=False):
        assert mode in ("off", "epoch", "batch")
        self.eta = eta
        self.mode = mode
        self.drift_bias = drift_bias
        self.multiplicative = multiplicative
    
    @torch.no_grad()
    def attach(self, model: nn.Module):
        # collect handles to drift-enabled layers: LinearRC and Conv2d
        self._targets = [m for m in model.modules() if isinstance(m, (LinearRC, nn.Conv2d))]

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
    def _apply_gaussian_nudge(self, model):
        for m in model.modules():
            if isinstance(m, LinearRC):
                if m.W is not None:
                    if self.multiplicative:
                        m.W.mul_(1.0 + torch.randn_like(m.W) * self.eta)
                    else:
                        m.W.add_(torch.randn_like(m.W) * self.eta)
                if m.b is not None and self.drift_bias:
                    if self.multiplicative:
                        m.b.mul_(1.0 + torch.randn_like(m.b) * self.eta)
                    else:
                        m.b.add_(torch.randn_like(m.b) * self.eta)
            elif isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    if self.multiplicative:
                        m.weight.mul_(1.0 + torch.randn_like(m.weight) * self.eta)
                    else:
                        m.weight.add_(torch.randn_like(m.weight) * self.eta)
                if m.bias is not None and self.drift_bias:
                    if self.multiplicative:
                        m.bias.mul_(1.0 + torch.randn_like(m.bias) * self.eta)
                    else:
                        m.bias.add_(torch.randn_like(m.bias) * self.eta)
