import os
import diffuser.utils as utils
import torch
from ml_logger import logger, RUN
from config.ct_hopper import Config

def main(**deps):
    RUN._update(deps)
    Config._update(deps)

    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)
    
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

    # Render
    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )
    renderer = render_config()

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
    model = cm_config(net=unet)

    # Loss
    loss_config = utils.Config(
        Config.loss,
        savepath='loss_config.pkl',
        action_dim=action_dim,
        sigma_min=Config.sigma_min,
        sigma_max=Config.sigma_max,
        sigma_data=Config.sigma_data,
        rho=Config.rho,
        loss_norm=Config.loss_norm,
        weight_schedule=Config.weight_schedule,
        curriculum=Config.curriculum,
        s0=Config.s0,
        s1=Config.s1,
        noise_schedule=Config.noise_schedule,
        Pmean=Config.Pmean,
        Pstd=Config.Pstd,
    )
    loss_fn = loss_config()

    # Trainer
    training_config = utils.Config(
        Config.trainer,
        savepath='trainer_config.pkl',
        K=Config.K,
        steps_per_epoch=Config.steps_per_epoch,
        batch_size=Config.batch_size,
        lr=Config.learning_rate,
        gradient_accumulate=Config.gradient_accumulate,
        step_update_ema=Config.step_update_ema,
        step_start_ema=Config.step_start_ema,
        ema_decay=Config.ema_decay,
        log_freq=Config.log_freq,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        n_reference=Config.n_reference,
        bucket=Config.bucket,
    )
    trainer = training_config(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        renderer=renderer,
        device=Config.device,
    )

    # Tests forward and backward passes
    utils.report_parameters(model)

    # Resume training
    loadpath = os.path.join(Config.bucket, logger.prefix, f'checkpoint/state.pt')
    if os.path.isfile(loadpath):
        trainer.load(loadpath)
    
    # Train
    trainer.train()
