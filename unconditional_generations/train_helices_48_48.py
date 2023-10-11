


import gc
import io
import time
from pathlib import Path
import concurrent.futures
from tqdm.notebook import tqdm
import h5py

import numpy as np
#import tensorflow as tf
import logging
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from training_utils import save_checkpoint, restore_checkpoint, create_model#, H5Dataset
from utils import save_grid
import datasets
import pickle as pkl
import matplotlib.pyplot as plt



from configs.config_ode48 import get_configs
config = get_configs(batch_size=72,size=48,num_ch=1) # previous batch size is 8. increased to 16.
#workdir = Path.cwd().joinpath("training","128_cath_s95_new_config")

workdir = Path.cwd().joinpath("training","helices_48_48_ch1")

num_train_steps = config.training.n_iters
batch_size = config.training.batch_size



class H5Dataset_modify(torch.utils.data.Dataset):
    def __init__(self, ds, device="cpu"):
        self.dataset = ds
        self.device = device


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]["data"]

        return data




import pickle 

with open("numpy_filter_rosettascore100_trainingdata_0512_48_48_for_protein_sgm.pkl","rb") as f:
    d1= pickle.load(f)
#with open("dataset_antibody_light_heavy_all_info_3500_7000_date0831.pkl","rb") as f:
#    d2= pickle.load(f)
#with open("dataset_antibody_light_heavy_all_info_7000_10768_date0831.pkl","rb") as f:
#    d3= pickle.load(f)


x =d1[10000:] #+ d2 +d3


length = int(len(x))

cut_p = int(length * 0.95)

dataset = H5Dataset_modify(x[:cut_p])

test_ds = H5Dataset_modify(x[cut_p:])


#from configs.config import get_configs
#config = get_configs(batch_size=8,size=128,num_ch=5)
#workdir = Path.cwd().joinpath("training","128_cath_s95_new_config_with_seq2_ode_only_oct10")

#num_train_steps = config.training.n_iters
#batch_size = config.training.batch_size


train_sampler = torch.utils.data.RandomSampler(
    dataset,
    replacement=True,
    num_samples=num_train_steps*batch_size
)
train_dl = torch.utils.data.DataLoader(
    dataset,
    sampler=train_sampler,
    batch_size=batch_size,
    drop_last=True
)

train_iter = iter(train_dl)

test_sampler = torch.utils.data.RandomSampler(
    test_ds,
    replacement=True,
    num_samples=num_train_steps*batch_size
)


test_dl = torch.utils.data.DataLoader(
    test_ds,
    sampler=test_sampler,
    batch_size=batch_size,
    drop_last=True
)
eval_iter = iter(test_dl)


# Create directories for experimental logs
workdir.mkdir(exist_ok=True)
sample_dir = workdir.joinpath("samples")
sample_dir.mkdir(exist_ok=True)

tb_dir = workdir.joinpath("tensorboard")
tb_dir.mkdir(exist_ok=True)
writer = tensorboard.SummaryWriter(tb_dir)

# Initialize model.
score_model = create_model(config)
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
optimizer = losses.get_optimizer(config, score_model.parameters())
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

# Create checkpoints directory
checkpoint_dir = workdir.joinpath("checkpoints")
# Intermediate checkpoints to resume training after pre-emption in cloud environments
checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_meta_dir.parent.mkdir(exist_ok=True)
# Resume training when intermediate checkpoints are detected
#state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
#initial_step = int(state['step'])
initial_step = 0
print(f"Starting from step {initial_step}...")

# Dummy scaler
inverse_scaler = datasets.get_dummy_scaler()

# Setup SDEs
if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

# Build one-step training and evaluation functions
optimize_fn = losses.optimization_manager(config)
continuous = config.training.continuous
reduce_mean = config.training.reduce_mean
likelihood_weighting = config.training.likelihood_weighting
train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                 reduce_mean=reduce_mean, continuous=continuous,
                                 likelihood_weighting=likelihood_weighting)
eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                reduce_mean=reduce_mean, continuous=continuous,
                                likelihood_weighting=likelihood_weighting)

# # # Building sampling functions
if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

for step in range(initial_step, num_train_steps + 1):
    batch = next(train_iter).to(config.device).float()
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
        writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
        save_checkpoint(checkpoint_meta_dir, state)

    # # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
        eval_batch = next(eval_iter).to(config.device).float()
        eval_loss = eval_step_fn(state, eval_batch)
        writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
        # Save the checkpoint.
        save_step = step // config.training.snapshot_freq
        save_checkpoint(checkpoint_dir.joinpath(f'checkpoint_{save_step}.pth'), state)

        # Generate and save samples
        if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = sample_dir.joinpath(f"iter_{step}")
            this_sample_dir.mkdir(exist_ok=True)

            with open(str(this_sample_dir.joinpath("sample.pkl")),"wb") as fout:
                pkl.dump(sample.cpu(),fout)

            save_grid(sample.cpu().numpy(),this_sample_dir.joinpath("sample.png"))
