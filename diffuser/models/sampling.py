import numpy as np
import torch
from diffuser.models.utils import pad_dims_like, apply_conditioning


@torch.no_grad()
def multistep_sampling(
    precond,
    shape,
    sigmas,  # sigmas[0] = sigma_min, sigmas[-1] = sigma_max
    returns,
    device,
    cond=None,
    action_dim=0,
):
    x = torch.randn(shape, device=device) * sigmas[-1]  # x ~ N(0, σ_max)
    sigma = torch.full((shape[0],), sigmas[-1], dtype=torch.float32, device=device)
    if cond is not None:
        x = apply_conditioning(x, cond, action_dim=action_dim)
    x = precond(x, sigma, returns, sampling=True)

    for sigma in reversed(sigmas[:-1]):
        if cond is not None:
            x = apply_conditioning(x, cond, action_dim=action_dim)
        sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
        x = x + torch.randn_like(x) * pad_dims_like((sigma**2 - sigmas[0]**2) ** (0.5), x) 
        x = precond(x, sigma, returns, sampling=True)

    if cond is not None:
        x = apply_conditioning(x, cond, action_dim=action_dim)
    return x