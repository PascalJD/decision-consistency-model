import torch
import torch.nn as nn
from diffuser.models.utils import mean_flat, pad_dims_like, reshape_sequences_and_actions
from diffuser.models.utils import karras_sigmas, sample_timestep, diff, get_weightings, get_numsteps

class CTLoss:
    """
    Loss function corresponding to the Consistency Training (CT) formulation
    """
    def __init__(
        self,
        action_dim,
        noise_schedule="CT+",
        Pmean=-1.1,
        Pstd=2.0,
        sigma_min=0.002, 
        sigma_max=80.0, 
        sigma_data=0.5, 
        rho=7.0,
        loss_norm='l2',
        weight_schedule="CT+",
        curriculum="CT+",
        s0=10,
        s1=1280,
        diffuse_action=False,
        device='cuda'
    ):
        self.action_dim = action_dim
        self.noise_schedule = noise_schedule
        self.Pmean = Pmean
        self.Pstd = Pstd
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.loss_norm = loss_norm
        self.weight_schedule = weight_schedule
        self.curriculum = curriculum
        self.s0 = s0
        self.s1 = s1
        self.diffuse_action = diffuse_action
        self.device = device

    def __call__(self, model, k, K, x, cond, returns=None):
        """
        loss from CT+ paper
        model = precond object  (Ema is removed ic CT+)
        k = current training step
        K = total number of training steps
        x ~ p_data(x)
        returns = conditionning

        returns the loss (batch_size,)
        """
        if not self.diffuse_action:
            x = x[:, :, self.action_dim:]  # x ~ p_data(x)
        batch_size = x.shape[0]
        z = torch.randn_like(x)  # z ~ N(0, I)

        N = get_numsteps(k, K, self.curriculum, self.s0, self.s1)
        sigmas = karras_sigmas(
            N, self.rho, self.sigma_min, self.sigma_max, self.device
        )
        i = sample_timestep(
            batch_size, sigmas, self.device, self.noise_schedule, self.Pmean, self.Pstd
        )
        sigma1 = sigmas[i]
        sigma2 = sigmas[i + 1]

        x2 = x + pad_dims_like(sigma2, x) * z  # x + σ_(i+1) * z  more noise
        denoise_student = model(x2, sigma2, returns)  # f_theta(x + σ_(i+1) * z, σ_(i+1))

        with torch.no_grad():
            x1 = x + pad_dims_like(sigma1, x) * z  # x + σ_i * z
            denoise_target = model(x1, sigma1, returns)  # f_theta-(x + σ_i * z, σ_i)   # EMA has less noise 

        d = diff(denoise_student, denoise_target, self.loss_norm)
        w = get_weightings(sigma1, sigma2, self.sigma_data, self.weight_schedule)

        loss = w * mean_flat(d)
        return loss.mean()


class ActionLoss:
    def __init__(
            self, 
            horizon,
            train_on_prediction=False,
        ):
        self.horizon = horizon
        self.train_on_prediction = train_on_prediction
        self.loss_fn = nn.MSELoss()

    def __call__(
            self, model, s, a
        ):
        """
        if train_on_prediction, s is sampled states from the model
        else s is the ground truth states from the training data
        """
        if self.train_on_prediction:  
            s = s[:, :self.horizon, :]
            a = a[:, 0, :]
        else:
            # append all sequences of length horizon to the batch size
            s, a = reshape_sequences_and_actions(s, a, self.horizon)
        
        a_pred = model(s)
        loss = self.loss_fn(a_pred, a)

        return loss
    

class ActionLoss_dd:
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def __call__(
            self, model, s, a
        ):
        s_t = s[:, :-1, :]
        s_tp1 = s[:, 1:, :]
        s_comb = torch.cat([s_t, s_tp1], dim=-1)
        s_comb = s_comb.reshape(-1, 2 * s.shape[-1])
        a_pred = model(s_comb)
        a_t = a[:, :-1, :]
        a_t = a_t.reshape(-1, a.shape[-1])
        loss = self.loss_fn(a_pred, a_t)
        return loss