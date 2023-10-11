# in total I will need at least two steps.

# first step, model from latent space  to get data space. this step is done in sampling0916.py

# second step, given a sample data. I need to generate the pdb format.
# Done. c_pdb from rosetta_do_mcmc.pu

# now merge.


from sko.SA import SA
  

# this part is to load saved model and an example to show generate sample.

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from training_utils import restore_checkpoint
from models.ema import ExponentialMovingAverage
from models import ncsnpp
import datasets
import sde_lib
import sampling_ode as sampling
import losses
import pickle as pkl
from tqdm.notebook import tqdm



from configs.config import get_configs
config = get_configs(batch_size=1,size=128,num_ch=5)
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


sampling_shape = (1, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


generated_samples = []
state['ema'].store(state["model"].parameters())
state['ema'].copy_to(state["model"].parameters())
#for i in tqdm(range(10)):
#    sample, n = sampling_fn(state["model"])
    # state['ema'].restore(state["model"].parameters())
#    generated_samples.append(sample.cpu())






#def prior_sampling(self, shape):
#    return torch.randn(*shape) * self.sigma_max




def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))



def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq



path_g1 = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/sampling/128_cath_s95_new_config_with_seq/tmp/"


rmsd_ratio = 10

target_pdb_name = "3JCX_1_H.pdb" 

name_f = target_pdb_name.split(".")[0]

kkk = 0

sigma_max= config.model.sigma_max

from align_generated_real_structures import grep_bb_from_pdb, grep_ca_array_from_bb_crds, cal_rmsd

from rosetta_do_for_mcmc import c_pdb

target_bb = grep_bb_from_pdb(target_pdb_name)

target_ca = grep_ca_array_from_bb_crds(target_bb)

sample_lst = []

def demo_func(fix_noise):# input batch size 1 
    #gc.collect()
    fix_noise = torch.from_numpy(fix_noise)
    fix_noise=fix_noise.float()
    #fix_noise = sigma_max * fix_noise
    # test what is goining on to use prior sampling in the orignal function
    samples, n = sampling_fn(state["model"],fix_noise)
    samples = samples.cpu()
    sample_lst.append(samples)
    sample = samples[0]
    #np_x = x.detach().numpy()
    np_sample = sample.detach().numpy()
    seq = get_seq(np_sample)
    c_pdb(sample,seq)
    g_pdb_name = path_g1 +"final_structure.pdb" 
    g_bb = grep_bb_from_pdb(g_pdb_name)
    g_ca = grep_ca_array_from_bb_crds(g_bb)
    rmsd=cal_rmsd(target_ca,g_ca)
    print(rmsd)
    return rmsd
    #return rmsd




import subprocess
def mcmc_val(loops):
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #ref_atom_name_lst =target_atom_names[kkk]
   # global kkk
   # kkk = kk
   # target_coords =np.array(target_2_res_lst[kkk],dtype="float32")

#    target_coords_NC = np.array([ca_lst[kkk][0]]+target_2_res_lst[kkk]+[ca_lst[kkk][-1]],dtype="float32")

#comb_num = len(target_coords)


   # saved_x = np.zeros(sampling_shape)
   # saved_y = 99999
    num = 0
    saved_lsts = []
    subprocess.run(['mkdir',path_g1+str(kkk)+"/"])
    for i in range(loops):
        #fix_noise = torch.randn(sampling_shape)
        fix_noise = sde.prior_sampling(sampling_shape)
        sa = SA(func=demo_func, x0=fix_noise, T_max=1, T_min=1e-1, L=10, max_stay_counter=5)
        best_x, best_y = sa.run()
        #saved_lsts.append(to_saved_lst)
        pickle_out_W = open(path_g1+str(kkk)+"_"+name_f+"_samples_lst.pickle","wb")
        pickle.dump(sample_lst,pickle_out_W)
        pickle_out_W.close()
        
        if best_y < rmsd_ratio:
            subprocess.run('cp',path_g1 +"final_structure.pdb",path_g1+str(kkk)+"/"+str(num)+".pdb")
            to_saved_lst = [best_x,best_y]
            saved_lsts.append(to_saved_lst)
            pickle_out_W = open(path_g1+str(kkk)+"_"+name_f+"_saved_lst_0217.pickle","wb")
            pickle.dump(saved_lsts,pickle_out_W)
            pickle_out_W.close()
            num+=1


mcmc_val(100)
