# utils.py


import numpy as np
import torch
import random

# - set_seed(seed: int)
# - to seed Python, NumPy, and PyTorch for reproducibility

def set_seed(seed: int) -> None:
    # Seed Python RNG
    random.seed(seed)
    # Seed NumPy
    np.random.seed(seed)
    # Seed PyTorch RNG, on CPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinsm, seeding GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# - device_auto()
# - If CUDA is available, use it, otherwise use CPU.

def device_auto():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
    
