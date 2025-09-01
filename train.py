# train.py 

import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F


# train_one_epoch
# - Runs a full pass through the training set

def train_one_epoch(model, loader, optimizer, device) -> float:
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

