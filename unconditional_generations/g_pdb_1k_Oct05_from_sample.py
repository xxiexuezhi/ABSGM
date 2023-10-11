# this is code for generating pdb regarding random generatd samples with latent space
# the purpose of this code is to make mcmc working


lst_f_names = ["lst_latent_pts_and_samples_240_one_hot_20_encoded_sep30_ode_only.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_21.pkl", "lst_latent_pts_and_samples_400_one_hot_20_encoded_sep30_ode_only_2.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_22.pkl", "lst_latent_pts_and_samples_400_one_hot_20_encoded_sep30_ode_only_4.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_25.pkl", "lst_latent_pts_and_samples_400_one_hot_20_encoded_sep30_ode_only_5.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_sep30_ode_only_10.pkl", "lst_latent_pts_and_samples_480_one_hot_20_encoded_sep30_ode_only_3.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_sep30_ode_only_12.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_16.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_sep30_ode_only_14.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_19.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_sep30_ode_only_15.pkl", "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_20.pkl"] 

lst_ff_names = ["lst_latent_pts_and_samples_320_one_hot_20_encoded_Oct10_ode_only_25_with_decrease_sigmamax100.pkl","lst_latent_pts_and_samples_5120_one_hot_20_encoded_Oct12_ode_only_25_with_decrease_sigmamax100_with_probability_flow_true_withbatch64.pkl","lst_latent_pts_and_samples_5120_one_hot_20_encoded_ode_only_25_with_decrease_sigmamax100_with_probability_flow_true_withbatch64_seq2_chekpoint34_oct_15_6.pkl"]


import sys,json,math,os
import tempfile
import numpy as np
from pathlib import Path
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import math
import gemmi
import py3Dmol
import uuid
from pyrosetta import *

import rosetta_min.run_fix_seq as run

kkk = 2
num = int(sys.argv[1])

#iter_name = "iter_" + folder_name + "0000"

DATASET_PATH = Path.cwd().joinpath(
        "sampling",
        "128_cath_s95_new_config_with_seq2_0916"
    )


p = Path("/home/xxie92/scratch/")

# read pickle here.

#f_name = "lst_latent_pts_and_samples_640_one_hot_20_encoded_Oct3_ode_only_25.pkl"

# udapte here on Oct 11 to run the new genreation using ode with sigmax 100.

#f_name = lst_f_names[kkk]
f_name = lst_ff_names[kkk]

the_path = "sampling/128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022/lst_latent_pts_and_samples_5120_one_hot_20_encoded_ode_only_with_probability_flow_true_withbatch64_seq2_heavy_only_chekpoint70_Jan_12_2023_1.pkl"

with open(the_path,"rb") as f:
    lst_samples = pkl.load(f)

samples = lst_samples[1]
print(samples.shape)
#with open(DATASET_PATH.joinpath("seq_lst_1000.pkl"),"rb") as f:
#    seq_lst = pkl.load(f)


# Extract appropriate batch from samples
#TASK_ID=int(sys.argv[1])
BATCH_SIZE=1000
#samples = torch.split(torch.tensor(samples),1)[TASK_ID-1].numpy()


#print(samples.shape)


#DATASET_NAME = iter_name

# Find length of sequence via mask

def c_pdb_with_index(sample,seq,j):
    msk = np.round(sample[4])

    L = 128

    #seq = "A"*L
    #seq = seq_lst[TASK_ID-1]
    npz={}
    for idx,name in enumerate(["dist","omega","theta","phi"]):
        npz[name] = np.clip(sample[idx].reshape(L,L),-1,1)

    # Inverse scaling
    npz["dist_abs"] = (npz["dist"]+1)*10
    npz["omega_abs"] = npz["omega"]*math.pi
    npz["theta_abs"] = npz["theta"]*math.pi
    npz["phi_abs"] = (npz["phi"]+1)*math.pi/2

    #outPath = DATASET_PATH.joinpath(
    outPath = p.joinpath(
        "10k_g_data_Jan_2023",
        f_name.split(".")[0],
        str(j)
    )

    n_iter = 1
    for n in range(n_iter):

        outPath_run = outPath
        #if outPath_run.joinpath("final_structure.pdb").is_file():
        #    continue

        _ = run.run_minimization(
            npz,
            seq,
            n_iter=n,
            scriptdir=Path.cwd().joinpath("rosetta_min"),
            outPath=outPath_run,
            angle_std=20, # Angular harmonic std
            dist_std=2 # Distance harmonic std
        )



def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))



def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq



def loop_samples(samples):
    l = len(samples)
    a = int(num * 100)
    b = int((num+1) * 100)
    for i in range(a,b):
        sample = samples[i]
    #np_x = x.detach().numpy()
        np_sample = sample.detach().numpy()
        seq = get_seq(np_sample)
        #print(seq)
        c_pdb_with_index(samples[i],seq,i)


loop_samples(samples)
