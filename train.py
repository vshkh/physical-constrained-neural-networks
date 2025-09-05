# train.py 

import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from constraints.drift import DriftController

# train_one_epoch
# - Runs a full pass through the training set

def train_one_epoch(model, loader, optimizer, device, drift_controller: Optional[DriftController] = None) -> float:
    # Indicdate the model is being trained.
    model.train()
    running_loss = 0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)   
        
        loss.backward()
        optimizer.step()
        if drift_controller is not None:
            drift_controller.step_batch(model)
        optimizer.zero_grad()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    # Indicate the model is being evaluated.
    model.eval()

    # Statistics to keep track of:
    running_loss = 0
    num_batches = 0
    num_correct = 0
    num_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        running_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1)
        num_correct += (preds == y).sum().item()
        num_samples += y.size(0)

    avg_loss = running_loss / max(1, num_batches)
    acc = num_correct / max(1, num_samples)

    return avg_loss, acc

# train.py
@torch.no_grad()
def evaluate_with_preds(model, loader, device):
    model.eval()
    running_loss = 0.0; num_batches = 0
    num_correct = 0;   num_samples = 0
    all_preds = [];    all_labels = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        running_loss += loss.item(); num_batches += 1
        preds = logits.argmax(dim=1)
        num_correct += (preds == y).sum().item()
        num_samples += y.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(y.detach().cpu())
    avg_loss = running_loss / max(1, num_batches)
    acc = num_correct / max(1, num_samples)
    import torch as _torch
    return avg_loss, acc, _torch.cat(all_preds), _torch.cat(all_labels)


