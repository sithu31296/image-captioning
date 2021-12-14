import torch
import numpy as np
import random
import time
import functools



def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6      # in M


def time(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time_sync()
        value = func(*args, **kwargs)
        toc = time_sync()
        elapsed = toc - tic
        print(f"Elapsed time: {elapsed * 1000:.2f}ms")
        return value
    return wrapper_timer