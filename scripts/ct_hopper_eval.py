import diffuser.utils as utils
from ml_logger import logger, RUN
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.ct_hopper import Config
from diffuser.utils.arrays import to_torch, to_np
from diffuser.models.utils import karras_sigmas
from diffuser.models.sampling import multistep_sampling
from d4rl.ope import normalize

def evaluate(**deps):
    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    # Data
    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
    )
    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    horizon = dataset.horizon
    # Render
    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )
    renderer = render_config()

    # Consistency model
    cm_data = torch.load("/home/pjutrasd/depot_symlink/projects/decision-consistency-model/out/CT/checkpoint/state.pt")
    # UNet
    unet_config = utils.Config(
        Config.unet,
        savepath='unet_config.pkl',
        transition_dim=observation_dim,
        dim=Config.dim,
        dim_mults=Config.dim_mults,
        condition_dropout=Config.condition_dropout,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )
    unet = unet_config()
    # CM
    cm_config = utils.Config(
        Config.cm_precond,
        savepath='model_config.pkl',
        returns_condition=Config.returns_condition,
        is_classifier_free_guided=Config.is_classifier_free_guided,
        guidance_w=Config.w,
        device=Config.device,
        sigma_min=Config.sigma_min,
        sigma_max=Config.sigma_max,
        sigma_data=Config.sigma_data,
        rho=Config.rho,
    )
    cm = cm_config(net=unet)
    cm.load_state_dict(cm_data['ema_model'])
    cm.requires_grad_(False)
    cm.eval()

    # Action model
    action_state_dict = torch.load(
        "/home/pjutrasd/depot_symlink/projects/decision-consistency-model/out/CT/action-model/checkpoint/state.pt", 
        map_location=Config.device
    )
    action_model_config = utils.Config(
        Config.action_net,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=Config.action_hidden_dim,
        horizon=Config.action_horizon,
        device=Config.device,
    )
    action_model = action_model_config()
    action_model.load_state_dict(action_state_dict['model'])
    action_model.requires_grad_(False)
    action_model.eval()

    num_eval = 5  # Number of parallel trajectories
    device = Config.device
    num_steps = 1

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]  # One env per eval
    dones = [0 for _ in range(num_eval)]  # boolean array 
    episode_rewards = [0 for _ in range(num_eval)]  # accumulating rewards

    t = 0
    obs_list = [env.reset()[None] for env in env_list]  # Initializing environments 
    obs = np.concatenate(obs_list, axis=0)                           
    recorded_obs = [deepcopy(obs[:, None])] 

    while sum(dones) <  num_eval:
        # State prediction
        obs = dataset.normalizer.normalize(obs, 'observations')
        cond = {0: to_torch(obs, device=device)}  # {t: val} condition on current observation (inpainting)
        sigmas = karras_sigmas(
            3, cm.rho, cm.sigma_min, cm.sigma_max, device
        )
        if cm.returns_condition:
            returns = Config.sampling_returns * torch.ones(num_eval, 1, device=device)  # Conditionning on returns=1 
        else:
            returns = None
        with torch.no_grad():
            samples = multistep_sampling(
                cm, [num_eval, horizon, observation_dim], sigmas, returns, device, cond
            ) 
        
        # Action prediction               
        action = action_model(samples)  
        
        samples_np = to_np(samples)
        action = to_np(action)
        action = dataset.normalizer.unnormalize(action, 'actions')

        # Save prediction at t=0
        if t == 0:
            normed_observations = samples_np[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            savepath = os.path.join('images', 'sample-planned.png')
            renderer.composite(savepath, observations)

        obs_list = []
        for i in range(num_eval):
            # Taking action
            # done if max path length is reached or env is unhealthy
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])  
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    # Save executed trajectories
    episode_rewards = np.array(episode_rewards)
    policy_id = Config.dataset[:-3]
    normalize_rewards = np.array([normalize(policy_id, reward) for reward in episode_rewards])
    print("diffusion steps: ", num_steps)
    print("episode rewards: ", episode_rewards)
    print("normalized rewards: ", normalize_rewards)
    print("average_ep_reward: ", np.mean(episode_rewards))
    print("std_ep_reward: ", np.std(episode_rewards))
    print("average_normalized_ep_reward: ", np.mean(normalize_rewards))
    print("std_normalized_ep_reward: ", np.std(normalize_rewards))

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})
    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite(savepath, recorded_obs)

    