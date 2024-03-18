import torch.nn as nn
import torch
from diffuser.models.utils import pad_dims_like, get_scalings


class CTPrecond(nn.Module):
    def __init__(
        self,
        net,
        returns_condition=True,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        is_classifier_free_guided=False,
        guidance_w=1.2
    ):
        super().__init__() 
        self.net = net 
        self.returns_condition = returns_condition
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.is_classifier_free_guided = is_classifier_free_guided
        self.guidance_w = guidance_w

    def forward(self, x, sigma, returns=None, sampling=False):
        c_skip, c_out, c_in, c_noise = get_scalings(sigma, self.sigma_min, self.sigma_data)
        c_skip = pad_dims_like(c_skip, x)
        c_out = pad_dims_like(c_out, x)
        # x = pad_dims_like(c_in, x) * x
        # sigma = c_noise

        if self.is_classifier_free_guided:
            if sampling:
                F_x_cond = self.net(x, sigma, returns, use_dropout=False)
                F_x_uncond = self.net(x, sigma, returns, force_dropout=True)
                F_x = (1 + self.guidance_w) * F_x_cond - self.guidance_w * F_x_uncond
            else:
                F_x = self.net(x, sigma, returns, use_dropout=True)
        elif self.returns_condition:
            F_x = self.net(x, sigma, returns, use_dropout=False)
        else:
            F_x = self.net(x, sigma)

        return c_out * F_x + c_skip * x
