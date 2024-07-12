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

import rosetta_min.run_fix_seq_inpaint as run


#folder_name = sys.argv[2]

#iter_name = "iter_" + folder_name + "0000"

DATASET_PATH = Path.cwd().joinpath(
        "sampling",
       # "128_cath_s95_new_config_with_seq"
       '128_cath_s95_new_config_with_seq2_0916'
    )



#with open(DATASET_PATH.joinpath("lst_latent_pts_and_samples_400_one_hot_20_encoded_sep30_ode_only_4.pkl"),"rb") as f:
#    lst_samples = pkl.load(f)

#samples = lst_samples[1]
#with open(DATASET_PATH.joinpath("seq_lst_1000.pkl"),"rb") as f:
#    seq_lst = pkl.load(f)


# Extract appropriate batch from samples
#TASK_ID=int(sys.argv[1])
BATCH_SIZE=1000
#samples = torch.split(torch.tensor(samples),1)[TASK_ID-1].numpy()


#print(samples.shape)


#DATASET_NAME = iter_name

# Find length of sequence via mask
import numpy as np

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))



def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq


def c_pdb(sample,seq):
    msk = np.round(sample[4])

   # L = 128
   # update L to len(seq) for unfixed seq
    L = len(seq)

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

    outPath = DATASET_PATH.joinpath(
        "tmp"
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



def c_pdb_with_path_inpaint(sample,seq,p,pdb_name,start_indice, end_indice): # adding the string p as path to saved the pdb files.

    msk = np.round(sample[4])

    #    L = 128
    # update L in to len(seq) for non 128 situations
    L = len(seq)
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


    p_path = Path(p)
    outPath = p_path

    n_iter = 1
    for n in range(n_iter):

        #outPath_run = outPath
        #if outPath_run.joinpath("final_structure.pdb").is_file():
        #    continue

        _ = run.run_minimization(
            pdb_name,
            npz,
            seq,
            scriptdir=Path.cwd().joinpath("rosetta_min"),
            outPath=outPath,
            start_indice=start_indice,
            end_indice = end_indice,
            angle_std=4.5,#20, # Angular harmonic std
            dist_std=0.02 # Distance harmonic std
        )

#sample = samples[211]
#np_sample = sample.detach().numpy()
#seq = get_seq(np_sample)


#c_pdb(sample,seq)
