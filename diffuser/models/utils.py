import torch
import math

def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

def apply_conditioning(x, conditions, action_dim):
    '''
    Inpainting
    '''
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()  
    return x


def pad_dims_like(x, other):
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def get_numsteps(k, K, curriculum="CT+", s0=10, s1=1280):
    """
    Computes N(k)
    """
    if curriculum == "CT+":
        K_prime = math.floor(
            K / (math.log2(math.floor(s1 / s0)) + 1)
        )
        N = s0 + math.pow(2, math.floor(k / K_prime))
        return min(N, s1) + 1
    elif curriculum == "square":
        return min(k * k, s1) + 1
    elif curriculum == "constant":
        return s1 + 1
    else:
        raise ValueError(f"Unknown curriculum {curriculum}")


def diff(input, target, norm='l2'):
    """
    Computes d(x,y)
    """
    if norm == 'l2':
        return (input - target) ** 2
    elif norm == 'l1':
        return torch.abs(input - target)
    elif norm == 'huber':
        c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
        return torch.sqrt((input - target) ** 2 + c ** 2) - c
    else:
        raise ValueError(f"Unknown loss norm {norm}")
    

def karras_sigmas(N, rho=7.0, sigma_min=0.002, sigma_max=80.0, device="cpu"):
    """
    sigmas from EDM paper
    """
    steps = torch.arange(N, device=device) / max(N - 1, 1)
    rho_inv = 1 / rho
    sigmas = sigma_min**rho_inv + steps * (sigma_max**rho_inv - sigma_min**rho_inv)
    sigmas = sigmas**rho
    return sigmas


def sample_timestep(batch_size, sigmas, device="cpu", noise_schedule="CT+", mean=-1.1, std=2.0):
    """
    Computes i ~ U[[1, N(k)]]
    """
    if noise_schedule == "CT+":
        return sample_lognormal_timestep(batch_size, sigmas, mean, std)
    elif noise_schedule == "CT":
        N = sigmas.shape[0]
        return sample_uniform_timestep(batch_size, N, device)
    else:
        raise ValueError(f"Unknown noise schedule {noise_schedule}")


def sample_uniform_timestep(batch_size, N, device="cpu"):
    """
    Computes i ~ U[[1, N(k)]]
    """
    return torch.randint(0, N - 1, (batch_size,), device=device)
    

def sample_lognormal_timestep(batch_size, sigmas, mean=-1.1, std=2.0):
    """
    Computes i \in [[1, N(k)]]
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()
    return torch.multinomial(pdf, batch_size, replacement=True)


def get_weightings(sigma1, sigma2, sigma_data, weight_schedule="CT+"):
    """
    Computes the weightings λ(σi)
    """
    if weight_schedule == "CT+":
        return 1 / (sigma2 - sigma1)
    elif weight_schedule == "CT":
        return 1
    elif weight_schedule == "Karras":
        return 1 / sigma1 ** 2 + 1 / sigma_data ** 2
    else:
        raise ValueError(f"Unknown weight schedule {weight_schedule}")
    

def get_scalings(sigma, sigma_min, sigma_data):
    """
    Satisfies boundary condition
    """
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2) 
    c_out = ((sigma - sigma_min) * sigma_data) / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
    c_noise = 1000 * 0.25 * torch.log(sigma + 1e-44)
    return c_skip, c_out, c_in, c_noise


def reshape_sequences_and_actions(s, a, horizon):
    # Assuming s shape is [batch_size, n, observation_dim]
    # Assuming a shape is [batch_size, n, action_dim]
    batch_size, n, observation_dim = s.shape
    
    # Initialize an empty list to hold the subsequences and corresponding actions
    subsequences = []
    actions = []
    
    for i in range(n - horizon + 1):
        # Extract the subsequence from all sequences in the batch
        subsequence = s[:, i:i + horizon, :]
        print(subsequence.shape)
        
        # Append to the list of subsequences
        subsequences.append(subsequence)
        
        # Extract the corresponding action
        action = a[:, i, :]
        
        # Append to the list of actions
        actions.append(action)
        
    # Concatenate along the batch dimension
    s_reshaped = torch.cat(subsequences, dim=0)
    a_reshaped = torch.cat(actions, dim=0)
    
    return s_reshaped, a_reshaped