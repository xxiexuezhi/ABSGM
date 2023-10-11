# in total I will need at least two steps.

# first step, model from latent space  to get data space. this step is done in sampling0916.py

# updated on first step oct 13. using sampling_ode_save_memory. which could gerneate one batch for 128 instead of 8. 

# second step, given a sample data. I need to generate the pdb format.
# second step is replaced by distance calculation to make the program possible running. previous version is too slow. 


# Done. c_pdb from rosetta_do_mcmc.pu

# now merge.


from sko.SA import SA
  
import sys
# this part is to load saved model and an example to show generate sample.

import numpy as np
import torch
from pathlib import Path
#import matplotlib.pyplot as plt
from training_utils import restore_checkpoint
from models.ema import ExponentialMovingAverage
from models import ncsnpp
import datasets
import sde_lib
import sampling_ode_save_memory as sampling
import losses
import pickle as pkl
from tqdm.notebook import tqdm

import pickle

from configs.config import get_configs
config = get_configs(batch_size=128,size=128,num_ch=5)
workdir = Path.cwd().joinpath("sampling","128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022")


#config.model.sigma_max = 100

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


sampling_shape = (32, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


#generated_samples = []
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



path_g1 = "/home/xxie92/scratch/proteinsgm/sampling/128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022/tmp/"


#rmsd_ratio = 10

distance_ratio = 56 #previous 57, 23hrs generated about 1g for each test case. #update back into 50 #update it into 55 only for heavey chain in index 0,2,4,5,6,8 #50 #40 forget to change last time # this is for eucudian distance btw encoded map of c-beta and c-beta

#target_pdb_name = "3JCX_1_H.pdb" 

#name_f = target_pdb_name.split(".")[0]

#kkk = 8
kkk = int(sys.argv[1])
sigma_max= config.model.sigma_max



with open("mcmc_testcases11_encoded_dataset_dict_Nov19_heavy_chain_only.pkl","rb") as f:
    test_d= pickle.load(f)


one_test = test_d[kkk]

test_name = one_test["name"]
name_f = test_name.split(".")[0]

target_test = one_test




to_save_lst = []



def cal_distance(one_test,samples):
    min_dis =9999
    test_name = one_test["name"]
    a_encode = one_test["data"][:1]
    for i in range(len(samples)):
        #a_encode = one_test["data"][:1]
        b_encode = samples[i][:1]
        distance = np.linalg.norm(a_encode-b_encode)
        if distance < min_dis:
            min_dis = distance
        if distance < distance_ratio:
            sam = samples[i].unsqueeze(0)
            to_save_lst.append(sam)    
    #saved_lst.append([test_name,min_rmsd, name])
    return min_dis



def demo_func(fix_noise):# input batch size 1 
    #gc.collect()
    fix_noise = torch.from_numpy(fix_noise)
    fix_noise=fix_noise.float()
    #fix_noise = sigma_max * fix_noise
    # test what is goining on to use prior sampling in the orignal function

    # this is updated due to the current input fix_noise dim changed into 1,5,64,64
    fix_noise = fix_noise[0]
    samples, n = sampling_fn(state["model"],fix_noise)
    samples = samples.cpu()
    #sample_lst.append(samples)
    #sample = samples[0]
    #np_x = x.detach().numpy()
    np_samples = samples.detach().numpy()
    #seq = get_seq(np_sample)
    #c_pdb(sample,seq)
    #g_pdb_name = path_g1 +"final_structure.pdb" 
    #g_bb = grep_bb_from_pdb(g_pdb_name)
    #g_ca = grep_ca_array_from_bb_crds(g_bb)
    #rmsd=cal_rmsd(target_ca,g_ca)
    dis = cal_distance(target_test,samples)
    print(dis)
    return dis
    #return rmsd



with open("pickle_mcmc_start_ptrs_shape_11_32_5_128_128.pkl","rb") as f:
    torch_ptr= pickle.load(f)


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
        #fix_noise[:32]=torch_ptr[kkk] # Nov 19. not used for heavy chain due to I didnt genreate. let's see random results first.
        ## updated on Oct 14th. the SA function seems only be able to take shape len1. over 1 seems not working. 
        ## I will reshape the fix_noise and update the demo func to select the first element.
        ## the content should be exactly same.
        fix_noise = fix_noise.unsqueeze(0) 
        sa = SA(func=demo_func, x0=fix_noise, T_max=1, T_min=0.7, L=3, max_stay_counter=3) # previous T_min = 1e-1
        best_x, best_y = sa.run()
        #saved_lsts.append(to_saved_lst)
        generated_samples = torch.cat(to_save_lst,0)
        pickle_out_W = open(path_g1+str(kkk)+"_"+name_f+"_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used_mcmc_on_heavychain_chain_only_nov_dec1_index_all_apr29.pickle","wb")
        pickle.dump(generated_samples,pickle_out_W)
        pickle_out_W.close()

# update on oct 27 .2022. the previous name _to_save_lst_lst_dis50_with_achor_ptrs_used.pickle is actually using dis40 instead of 50. due to my mistake. 
mcmc_val(10000000)
