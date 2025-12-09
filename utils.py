import time
import torch


def profile_block(name, func, *args, **kwargs):
    out = func(*args, **kwargs)
    return out

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"{name} Time: {t1 - t0:.6f} seconds")
    return out
