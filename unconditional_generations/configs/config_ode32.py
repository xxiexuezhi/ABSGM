import ml_collections
import torch

def get_configs(batch_size=16,size=32,num_ch=2): # batch size previously is 16. I update into 8
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = batch_size
    training.sde = "vesde"
    training.n_iters = 1300001
    training.snapshot_freq = 10000
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = True  # previous is False. for ode, change it into true.
    sampling.snr = 0.17
    sampling.method = 'pc' # this value previous is pc. i change it into ode
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 5000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'sidechainnet'
    data.image_size = size
    data.random_flip = False
    # data.uniform_dequantization = False
    data.centered = False
    data.num_channels = num_ch
    # data.category = 'bedroom'

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 100.
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'positional'
    
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 32
    model.ch_mult = (1, 1, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return config
