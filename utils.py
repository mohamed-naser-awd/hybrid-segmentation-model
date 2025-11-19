import time
import torch


def profile_block(name, func, *args, **kwargs):
    out = func(*args, **kwargs)
    return out

    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    t1 = time.perf_counter()
    print(f"{name} Time: {t1 - t0:.6f} seconds")
    return out

    if any(isinstance(a, torch.Tensor) and a.is_cuda for a in args):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    if any(isinstance(a, torch.Tensor) and a.is_cuda for a in args):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"{name} Time: {t1 - t0:.6f} seconds")
    return out
