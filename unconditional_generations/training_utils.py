import torch
#import tensorflow as tf
import os

import logging
import torch
from models import ncsnpp
import h5py

def restore_checkpoint(ckpt_dir, state, device):
    if True:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def create_model(config):
    """Create the score model."""
    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model

# Removed train/test split since it's arbitrary in generative model evaluation
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path,seed=42):
        super().__init__()
        torch.manual_seed(seed)
        h5_file = h5py.File(file_path , 'r')
        self.data = h5_file['data']
        self.length = self.data.shape[0]
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length
