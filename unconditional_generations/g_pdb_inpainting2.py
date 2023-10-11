from align_generated_real_structures import grep_bb_from_pdb, grep_ca_array_from_bb_crds, cal_rmsd

from rosetta_do_for_mcmc import c_pdb_with_path

import pickle

import numpy
import numpy as np

import numpy
import numpy as np
import torch

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))


import subprocess

def cp_pdb(pre,after):
    subprocess.call(["cp", pre, after])


def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq


f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/rosetta_ab_benchmark/samples_0.pkl"

with open(f_name,"rb") as f:
    samples= pickle.load(f)


def g1_for_inpaint(j):
    #subprocess.call(["mkdir",path_g2])
    #subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    #sample = samples[i][0] # this part is updated on nov 1 2022 due to in saved memory form, there is no distance
    sample = samples[j]
        # this will only work for inpainting module
    #new_sample = torch.ones([5,128,128])
    path_g2="tmp/"

    
    np_sample = sample.detach().numpy()
    seq = get_seq(np_sample)
    print(seq)
    c_pdb_with_path(np_sample,seq,path_g2)



def g_all_for_inpaint(sample_index): #sample_index is for sample name. it is not the same as j in previous function.
    
    f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/sas_benchmark/samples_"+str(sample_index)+".pkl"

    with open(f_name,"rb") as f:
        samples= pickle.load(f)

    #subprocess.call(["mkdir",path_g2])
    #subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    #sample = samples[i][0] # this part is updated on nov 1 2022 due to in saved memory form, there is no distance
    for j in range(len(samples)):
        sample = samples[j]
        # this will only work for inpainting module
    #new_sample = torch.ones([5,128,128])
        path_g2="tmp/"+ str(sample_index)+"/"

        pre = path_g2
        np_sample = sample.detach().numpy()
        seq = get_seq(np_sample)
        print(seq)
        c_pdb_with_path(np_sample,seq,path_g2)
        pre = path_g2 
        cp_pdb(pre+"final_structure.pdb", pre+"sas0_"+str(sample_index)+"_"+str(j)+".pdb")
#g1_for_inpaint(0)

import sys

index = int(sys.argv[1])

g_all_for_inpaint(index)

#for index in range(6):
#    g_all_for_inpaint(index)
