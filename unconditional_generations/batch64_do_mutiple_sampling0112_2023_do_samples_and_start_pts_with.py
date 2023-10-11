import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from training_utils import restore_checkpoint
from models.ema import ExponentialMovingAverage
from models import ncsnpp
import datasets
import sde_lib
import sampling_ode_save_memory as sampling
import losses
import pickle as pkl
from tqdm.notebook import tqdm



from configs.config import get_configs
config = get_configs(batch_size=16,size=128,num_ch=5)
workdir = Path.cwd().joinpath("sampling","128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022")


# previous is 200. I decreased into 100 to see the difference.
# I dont plan to change on heavy chain only generations.
#config.model.sigma_max = 100

#config.sampling.probability_flow = True

checkpoint_dir = Path.cwd().joinpath(
    "training",
    "128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022",
    "checkpoints",
    "checkpoint_70.pth"
)


device = config.device


def create_model(config):
    """Create the score model."""
    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model


 # Initialize model.
score_model = create_model(config)
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
optimizer = losses.get_optimizer(config, score_model.parameters())
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)


loaded_state = torch.load(checkpoint_dir, map_location=device)
state['optimizer'].load_state_dict(loaded_state['optimizer'])
state['model'].load_state_dict(loaded_state['model'], strict=False)
state['ema'].load_state_dict(loaded_state['ema'])
state['step'] = loaded_state['step']

del loaded_state


inverse_scaler = datasets.get_dummy_scaler()
if config.training.sde == "vesde":
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
elif config.training.sde == "vpsde":
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3


sampling_shape = (64, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


generated_samples = []
latent_lst = []
state['ema'].store(state["model"].parameters())
state['ema'].copy_to(state["model"].parameters())
for i in tqdm(range(80)):
    fix_noise = sde.prior_sampling(sampling_shape)
    latent_lst.append(fix_noise)
    sample, n = sampling_fn(state["model"],fix_noise)
    # state['ema'].restore(state["model"].parameters())
    generated_samples.append(sample.cpu())

generated_samples = torch.cat(generated_samples,0)
latent_l = torch.cat(latent_lst,0)

workdir.mkdir(parents=True,exist_ok=True)
with open(workdir.joinpath("lst_latent_pts_and_samples_5120_one_hot_20_encoded_ode_only_with_probability_flow_true_withbatch64_seq2_heavy_only_chekpoint70_Jan_12_2023_5.pkl"), "wb") as f:
    pkl.dump([latent_l,generated_samples],f)
