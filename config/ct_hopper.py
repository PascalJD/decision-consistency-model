import torch
from params_proto.neo_proto import ParamsProto

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/home/pjutrasd/depot_symlink/projects/decision-consistency-model/out'
    dataset = 'hopper-medium-expert-v2'
    renderer = 'utils.MuJoCoRenderer'

    # Data
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    horizon = 100
    discount = 0.99
    max_path_length = 1000
    termination_penalty = -100
    returns_scale = 400.0

    # Unet
    unet = 'models.TemporalUnet'
    dim_mults = (1, 4, 8)
    condition_dropout = 0.25
    dim = 128

    # CT Preconditioning & Loss
    cm_precond = 'models.CTPrecond'
    loss = 'models.CTLoss'
    loss_norm = "huber"
    noise_schedule= "CT+"
    weight_schedule="CT+"
    curriculum="CT+"
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_data = 0.5
    rho=7
    s0 = 10
    s1 = 1280  
    Pmean = -1.1
    Pstd = 2.0
    returns_condition = True
    is_classifier_free_guided = True
    load_path = "/home/pjutrasd/depot_symlink/projects/decision-consistency-model/out/CT/checkpoint/state.pt"

    # Training
    trainer = 'models.CTTrainer'
    K = 1e6
    steps_per_epoch = 10000    
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate = 2

    step_update_ema = 10
    step_start_ema = 2000
    ema_decay = 0.995

    log_freq = 1000
    save_freq = 10000
    sample_freq = 2000
    n_reference = 4

    # Sampling
    sampling_returns = 0.9
    w = 1.2

    # Action model
    action_net = 'models.ActionCNN'
    action_hidden_dim = 512
    action_horizon = 100
    action_loss = 'models.ActionLoss'
    action_trainer = 'models.ActionTrainer'
    action_K = 1e5
    action_batch_size = 32
    action_learning_rate = 2e-4
    action_gradient_accumulate = 2
    action_log_freq = 1000
    action_save_freq = 10000
