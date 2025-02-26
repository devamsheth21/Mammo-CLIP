import os
import numpy as np
import pandas as pd
import random
import torch
\
def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)