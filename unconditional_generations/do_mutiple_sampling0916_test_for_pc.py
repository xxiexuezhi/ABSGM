import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from training_utils import restore_checkpoint
from models.ema import ExponentialMovingAverage
from models import ncsnpp
import datasets
import sde_lib
import sampling as sampling
import losses
import pickle as pkl
from tqdm.notebook import tqdm



from configs.config import get_configs
config = get_configs(batch_size=16,size=128,num_ch=5)
workdir = Path.cwd().joinpath("sampling","128_cath_s95_new_config_with_seq2_0916")


config.model.sigma_max = 200

checkpoint_dir = Path.cwd().joinpath(
    "training",
    "128_cath_s95_new_config_with_seq2",
    "checkpoints",
    "checkpoint_34.pth"
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


sampling_shape = (50, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


generated_samples = []
state['ema'].store(state["model"].parameters())
state['ema'].copy_to(state["model"].parameters())
for i in tqdm(range(10)):
    sample, n = sampling_fn(state["model"])
    # state['ema'].restore(state["model"].parameters())
    generated_samples.append(sample.cpu())

generated_samples = torch.cat(generated_samples,0)

workdir.mkdir(parents=True,exist_ok=True)
with open(workdir.joinpath("samples_1k_one_hot_20_encoded_sep27_ode_only_for_pc.pkl"), "wb") as f:
    pkl.dump(generated_samples,f)
