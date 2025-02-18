import random
import numpy as np
import torch

def set_seed(seed=17):
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch (CPU)
    
    # I don't think I'm using CUDA but this should be quick and easy to set so whatever
    torch.cuda.manual_seed(seed)  # PyTorch (CUDA)
    torch.cuda.manual_seed_all(seed)  # PyTorch (all GPU devices)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CNNs
    torch.backends.cudnn.benchmark = False  # Disable auto-optimization for deterministic runs

    print(f"Global seed set to {seed}")
