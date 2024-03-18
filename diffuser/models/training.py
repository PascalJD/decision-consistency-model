import os
import copy
import torch
from diffuser.utils.timer import Timer
from diffuser.utils.arrays import batch_to_device, to_np
from ml_logger import logger
from diffuser.models.sampling import multistep_sampling
from diffuser.models.utils import karras_sigmas


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class CTTrainer():
    def __init__(
        self,
        dataset,
        model,
        loss_fn,
        renderer,
        action_model=None,  
        action_loss_fn=None,
        step_update_ema = 10,
        step_start_ema = 2000,
        ema_decay = 0.995,
        K=1e6,
        steps_per_epoch=10000,
        batch_size=32,
        lr=2e-5,
        gradient_accumulate=2,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        n_reference=4,
        bucket=None,
        device="cuda",
    ):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.renderer = renderer
        self.action_model = action_model
        self.action_loss_fn = action_loss_fn

        # EMA for student to achieve better sample quality at inference time
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.step_update_ema = step_update_ema
        self.step_start_ema = step_start_ema

        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.gradient_accumulate = gradient_accumulate

        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.n_reference = n_reference

        self.bucket = bucket
        self.device = device

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(  # For visualization
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))

        if self.action_model is not None:
            params = list(self.action_model.parameters()) + list(self.model.parameters())
        else:
            params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.k = 0

    def reset_ema(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_ema(self):
        if self.k < self.step_start_ema:
            self.reset_ema()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self):
        timer = Timer()
        n_epochs = int(self.K // self.steps_per_epoch)

        while self.k <= self.K:
            # Train step
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulate):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss = self.loss_fn(self.model, self.k, self.K, *batch).mean()
                if self.action_model is not None:
                    x, cond, returns = batch
                    s = x[:, :, self.action_model.action_dim:]   # ground truth states
                    a = x[:, :, :self.action_model.action_dim]   # ground truth actions
                    action_loss = self.action_loss_fn(self.action_model, s, a).mean()
                    loss += action_loss
                    loss = loss / 2  # Average diffuse and action losses
                loss = loss.mul(1 / self.gradient_accumulate)
                loss.backward()
            self.optimizer.step()

            # Update EMA
            if self.k % self.step_update_ema == 0:
                self.update_ema()

            # Save
            if self.k % self.save_freq == 0:
                self.save()

            # Log
            if self.k % self.log_freq == 0:
                metrics = {}
                metrics['time'] = timer()
                metrics['step'] = self.k
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics)
            if self.k % self.steps_per_epoch == 0:
                logger.print(f'Epoch {self.k // self.steps_per_epoch} / {n_epochs} | {logger.prefix}')

            # Sample
            # if self.k == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)
            if self.sample_freq and self.k % self.sample_freq == 0:
                self.render_samples()

            self.k += 1

    def save(self):
        '''
        saves model, ema, and step;
        syncs to storage bucket if a bucket is specified
        '''
        data = {
            'k': self.k,
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
        }
        if self.action_model is not None:
            data['action_model'] = self.action_model.state_dict()
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ models/training ] Saved model to {savepath}')

    def load(self, loadpath=None):
        '''
        loads student, ema_student, target, and step;
        '''
        if loadpath is None:
            loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')

        data = torch.load(loadpath)

        self.k = data['k']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema_model'])
        if self.action_model is not None:
            self.action_model.load_state_dict(data['action_model'])
        logger.print(f'[ models/training ] Loaded model from {loadpath}')

    def render_reference(self, batch_size=10):
        '''
        renders a batch of training points
        '''
        # temporary dataloader
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        # get trajectories and condition at t=0
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        # [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=4):
        '''
        Renders samples from EMA model (states only)
        '''
        if self.model.returns_condition:
            returns = torch.ones((batch_size,1), device=self.device)  # Conditionning on returns=1 
        else:
            returns = None

        horizon = self.dataset.horizon
        transition_dim = self.model.net.transition_dim
        shape = (batch_size, horizon, transition_dim)

        sigmas = karras_sigmas(
            3, self.loss_fn.rho, self.loss_fn.sigma_min, self.loss_fn.sigma_max, self.device
        )

        samples = multistep_sampling(
            self.ema_model, shape, sigmas, returns, self.device
        )
        if self.loss_fn.diffuse_action:
            samples = samples[:, :, self.dataset.action_dim:]
        samples = samples.detach().cpu().numpy()
        observations = self.dataset.normalizer.unnormalize(samples, 'observations')

        savepath = os.path.join('images', f'train_sample.png')
        self.renderer.composite(savepath, observations)


class ActionTrainer():
    def __init__(
        self,
        dataset,
        model,
        loss_fn,
        cm=None,
        K=1e6,
        steps_per_epoch=10000,
        batch_size=32,
        lr=2e-5,
        gradient_accumulate=2,
        log_freq=100,
        bucket=None,
        train_device="cuda",
        save_freq=1000,
        save_checkpoint=False,
    ):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        
        self.cm = cm

        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.gradient_accumulate = gradient_accumulate

        self.log_freq = log_freq
        self.bucket = bucket
        self.train_device = train_device
        self.save_freq = save_freq
        self.save_checkpoint = save_checkpoint

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.k = 0

    def train(self):
        timer = Timer()
        n_epochs = int(self.K // self.steps_per_epoch)

        while self.k <= self.K:
            # Train step
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulate):
                batch = next(self.dataloader)
                x, cond, returns = batch_to_device(batch, device=self.train_device)
                s = x[:, :, self.model.action_dim:] #ground truth state
                a = x[:, :, :self.model.action_dim] # ground truth action
                if self.cm is not None:
                    sigmas = karras_sigmas(
                        3, self.cm.rho, self.cm.sigma_min, self.cm.sigma_max, self.train_device
                    )
                    if self.cm.returns_condition:
                        returns = torch.ones((x.shape[0],1), device=self.train_device)  # Conditionning on returns=1 
                    else:
                        returns = None
                    with torch.no_grad():
                        s = multistep_sampling(
                            self.cm, s.shape, sigmas, returns, self.train_device, cond
                        ) 
                loss = self.loss_fn(self.model, s, a).mean()
                loss = loss.mul(1 / self.gradient_accumulate)
                loss.backward()
            self.optimizer.step()
            

            # Save
            if self.k % self.save_freq == 0:
                self.save()

            # Log
            if self.k % self.log_freq == 0:
                metrics = {}
                metrics['time'] = timer()
                metrics['step'] = self.k
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics)
            if self.k % self.steps_per_epoch == 0:
                logger.print(f'Epoch {self.k // self.steps_per_epoch} / {n_epochs} | {logger.prefix}')

            self.k += 1

    def save(self):
        data = {
            'step': self.k,
            'model': self.model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ models/training ] Saved model to {savepath}')

    def load(self, loadpath=None):
        if loadpath is None:
            loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')

        data = torch.load(loadpath)

        self.k = data['step']
        self.model.load_state_dict(data['model'])
        logger.print(f'[ models/training ] Loaded model from {loadpath}')