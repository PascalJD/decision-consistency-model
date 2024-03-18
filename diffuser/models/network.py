"""
Based on https://github.com/anuragajay/decision-diffuser/tree/main/code
"""
import math
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_positions=10000, endpoint=False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.dim//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    

# ----------------------------------------------------------------------------
# Decision Diffuser Temporal Unet from papers
# "Is Conditional Generative Modeling All You Need For Decision Making?" and
# "Planning with Diffusion for Flexible Behavior Synthesis".

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        act_fn = nn.Mish()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        act_fn = nn.Mish()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, t):
        """
        x : [ batch_size x in_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.skip_connection(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        kernel_size=5,
        # is_cm=False,
    ):
        super().__init__()
        self.time_dim = dim
        self.returns_dim = dim
        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.transition_dim = transition_dim
        # self.is_cm = is_cm

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        act_fn = nn.Mish()

        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        # Condition embedding
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim
        # if self.is_cm:
        #     embed_dim += dim

        num_resolutions = len(in_out)

        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Middle block
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size
        )

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=embed_dim, kernel_size=kernel_size
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        """
        x : [ batch x horizon x transition ]
        returns : [batch x horizon]
        """

        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)  # conditionned if use_dropout=False and force_dropout=False
            if use_dropout:  # drop condition during training according to probability condition_dropout
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:  # unconditionned (for classifier free sampling)
                returns_embed = 0 * returns_embed            
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")

        return x


class ActionMLP_dd(nn.Module):
    """
    2-layered MLP
    f(s_t, s_{t+1}) = a_t
    """
    def __init__(self, observation_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, x):
        return self.inv_model(x)
    


class ActionCNN(nn.Module):
    """
    1D Convolutional Neural Network
    f(s_t, s_{t+1}, ..., s_{t+horizon}) = a_t
    """
    def __init__(self, observation_dim, action_dim, hidden_dim=256, horizon=100, num_groups=4):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_groups = num_groups

        self.inv_model = nn.Sequential(
            nn.Conv1d(self.observation_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * self.horizon, self.action_dim),
        )

    def forward(self, x):
        # Assuming input shape is (batch_size, sequence_length, observation_dim)
        x = x.permute(0, 2, 1) # Change shape to (batch_size, observation_dim, sequence_length)
        return self.inv_model(x)





