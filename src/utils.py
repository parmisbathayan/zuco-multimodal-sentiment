"""Small filesystem and reproducibility helpers."""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_run_tag():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def save_json(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)
