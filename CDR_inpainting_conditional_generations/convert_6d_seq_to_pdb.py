  import numpy as np

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))








from pathlib import Path

import subprocess

def cp_pdb(pre,after):
    subprocess.call(["cp", pre, after])


def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq



from abnumber import Chain

# no_index version
def get_h1_h2_h3(seq):
    try:
        chain = Chain(seq, scheme='chothia')
        return chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq
    except:
        print(i)

def get_pos(cdr,seq):
    return [seq.find(cdr),seq.find(cdr)+len(cdr)]

def get_h1_2_3_pos_lst(seq):
    full_seq = seq
    chain = Chain(seq, scheme='chothia')
    h1,h2,h3 = chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq
    #cdr =
    return get_pos(h1,full_seq), get_pos(h2,full_seq), get_pos(h3,full_seq)


import gemmi


def get_seq_from_pdb(p):
    st = gemmi.read_structure(str(p))
    st.setup_entities()  
    polymer = st[0][0].get_polymer()
    sequence = gemmi.one_letter_code(polymer)
    return sequence




import sys
#the_num = int(sys.argv[2])



cdr_num = int(sys.argv[4])

from rosetta_do_for_inpaint import c_pdb_with_path_inpaint

import pickle

import numpy
import numpy as np

import numpy
import numpy as np
import torch




def g_all_for_inpaint(f_sample_pkl,ref_pdb_name,sample_index): # this function is updated with given length #sample_index is for sample name. it is not the same as j in previous function.
    
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/rosetta_ab_benchmark/"+str(the_num)+"/"+"samples_"+str(sample_index)+".pkl"
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/rosetta_ab_with_epitope/checkpoints/checkpoint_"+str(the_num)+"/rosetta_antibody_benchmark_epitope_antibody_mar15/"+"samples_"+str(sample_index)+".pkl"
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-inpaint-benchmark/training/inpainting_ch6/checkpoint_"+str(the_num)+"/h3_sim_less50_benchmark_epitope_antibody_mar27/"+"samples_"+str(sample_index)+".pkl"
    #L = len(ref_lst2[sample_index])
    #f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-inpaint-benchmark/training/single_cdr/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h"+str(cdr_num)+"_mar29/"+"samples_"+str(sample_index)+".pkl"
    
    #f_name = "/home/xxie92/scratch/protein-sgm-inpaint-benchmark/training/single_cdr_apr18/inpainting_ch6/checkpoint_"+str(the_num)+"/h3_sim_less50_benchmark_epitope_antibody_h"+str(cdr_num)+"_only_training_apr18/"+"samples_"+str(sample_index)+".pkl"
    #f_name = "/home/xxie92/scratch/backup2/protein-sgm-inpaint-benchmark/training/single_cdr_jun26/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h1_Jun22/"+"samples_"+str(sample_index)+".pkl"
    
    #f_name ="./training/single_cdr_jun26/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h1_Jun22/"+"samples_"+str(sample_index)+".pkl" 
    
    f_name = f_sample_pkl
    g_dir = "./test_single_cdr_inpaint_generations_single_h"+str(cdr_num)+"/"
    #g_dir ="/home/xxie92/scratch/backup2/test_jun27_single_cdr_inpaint_generations_single_h"+str(cdr_num)+"_checkptr"+str(the_num)+"_only_jun25_half_h"+str(cdr_num)+"/"
    p = Path(g_dir)
    p.mkdir(exist_ok=True,parents=True)
    #g_dir = g_dir + str(cdr_num)+"/"
    with open(f_name,"rb") as f:
        samples= pickle.load(f)
    
    # start herhe


    pdb_name = ref_pdb_name  #"half_h3_single_heavy_benckmark_pdb/" + pdb_name_lst[sample_index]
    

    ref_seq = get_seq_from_pdb(pdb_name)

    L = len(ref_seq)

    indice_lst = get_h1_2_3_pos_lst(ref_seq)

    start_indice = indice_lst[cdr_num-1][0]
    end_indice = indice_lst[cdr_num-1][1]






    # L is the length for the single heavychain. we will merge all the antigen pdb files separately. 

    samples = samples[:,:5,:L,:L]  # SO, b, c , n , n
    #subprocess.call(["mkdir",path_g2])
    #subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    #sample = samples[i][0] # this part is updated on nov 1 2022 due to in saved memory form, there is no distance

    #pdb_id = pdb_name.split("/")[-1][:-4] 

    

    for j in range(len(samples)):
        sample = samples[j]
        # this will only work for inpainting module
    #new_sample = torch.ones([5,128,128])
        path_g2= g_dir+str(sample_index)+"/"
        pre3 = g_dir
        pre = path_g2
        np_sample = sample.detach().numpy()
        seq = get_seq(np_sample)
        #print(seq)
        #c_pdb_with_path(np_sample,seq,path_g2)
        print(np_sample,seq,path_g2,pdb_name,start_indice, end_indice)

        c_pdb_with_path_inpaint(np_sample,seq,path_g2,pdb_name,start_indice, end_indice)
        pre = path_g2 
        cp_pdb(pre+"final_structure.pdb", pre3+"b"+"_"+str(sample_index)+"_"+str(j)+".pdb")
#g1_for_inpaint(0)

import sys

#index = int(sys.argv[1])
f_sample_pkl = sys.argv[1]

ref_pdb_name = sys.argv[2]

sample_index = int(sys.argv[3])

g_all_for_inpaint(f_sample_pkl,ref_pdb_name,sample_index)

#g_all_for_inpaint(index)

(base) [xxie92@beluga2 6d_to_pdb_inpaint]$ cat rosetta_do_for_inpaint.py 
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
(base) [xxie92@beluga2 6d_to_pdb_inpaint]$ cat convert_6d_seq_to_pdb.py 
import numpy as np

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


#print(one_hot_decode(np_x[0][4][:,64-10:64+11]))








from pathlib import Path

import subprocess

def cp_pdb(pre,after):
    subprocess.call(["cp", pre, after])


def get_seq(sample):
    seq = one_hot_decode(sample[4][:,64-10:64+10])
    return seq



from abnumber import Chain

# no_index version
def get_h1_h2_h3(seq):
    try:
        chain = Chain(seq, scheme='chothia')
        return chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq
    except:
        print(i)

def get_pos(cdr,seq):
    return [seq.find(cdr),seq.find(cdr)+len(cdr)]

def get_h1_2_3_pos_lst(seq):
    full_seq = seq
    chain = Chain(seq, scheme='chothia')
    h1,h2,h3 = chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq
    #cdr =
    return get_pos(h1,full_seq), get_pos(h2,full_seq), get_pos(h3,full_seq)


import gemmi


def get_seq_from_pdb(p):
    st = gemmi.read_structure(str(p))
    st.setup_entities()  
    polymer = st[0][0].get_polymer()
    sequence = gemmi.one_letter_code(polymer)
    return sequence




import sys
#the_num = int(sys.argv[2])



cdr_num = int(sys.argv[4])

from rosetta_do_for_inpaint import c_pdb_with_path_inpaint

import pickle

import numpy
import numpy as np

import numpy
import numpy as np
import torch




def g_all_for_inpaint(f_sample_pkl,ref_pdb_name,sample_index): # this function is updated with given length #sample_index is for sample name. it is not the same as j in previous function.
    
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/rosetta_ab_benchmark/"+str(the_num)+"/"+"samples_"+str(sample_index)+".pkl"
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/rosetta_ab_with_epitope/checkpoints/checkpoint_"+str(the_num)+"/rosetta_antibody_benchmark_epitope_antibody_mar15/"+"samples_"+str(sample_index)+".pkl"
   # f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-inpaint-benchmark/training/inpainting_ch6/checkpoint_"+str(the_num)+"/h3_sim_less50_benchmark_epitope_antibody_mar27/"+"samples_"+str(sample_index)+".pkl"
    #L = len(ref_lst2[sample_index])
    #f_name = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-inpaint-benchmark/training/single_cdr/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h"+str(cdr_num)+"_mar29/"+"samples_"+str(sample_index)+".pkl"
    
    #f_name = "/home/xxie92/scratch/protein-sgm-inpaint-benchmark/training/single_cdr_apr18/inpainting_ch6/checkpoint_"+str(the_num)+"/h3_sim_less50_benchmark_epitope_antibody_h"+str(cdr_num)+"_only_training_apr18/"+"samples_"+str(sample_index)+".pkl"
    #f_name = "/home/xxie92/scratch/backup2/protein-sgm-inpaint-benchmark/training/single_cdr_jun26/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h1_Jun22/"+"samples_"+str(sample_index)+".pkl"
    
    #f_name ="./training/single_cdr_jun26/inpainting_ch6/checkpoint_"+str(the_num)+"/masked_h1_Jun22/"+"samples_"+str(sample_index)+".pkl" 
    
    f_name = f_sample_pkl
    g_dir = "./test_single_cdr_inpaint_generations_single_h"+str(cdr_num)+"/"
    #g_dir ="/home/xxie92/scratch/backup2/test_jun27_single_cdr_inpaint_generations_single_h"+str(cdr_num)+"_checkptr"+str(the_num)+"_only_jun25_half_h"+str(cdr_num)+"/"
    p = Path(g_dir)
    p.mkdir(exist_ok=True,parents=True)
    #g_dir = g_dir + str(cdr_num)+"/"
    with open(f_name,"rb") as f:
        samples= pickle.load(f)
    
    # start herhe


    pdb_name = ref_pdb_name  #"half_h3_single_heavy_benckmark_pdb/" + pdb_name_lst[sample_index]
    

    ref_seq = get_seq_from_pdb(pdb_name)

    L = len(ref_seq)

    indice_lst = get_h1_2_3_pos_lst(ref_seq)

    start_indice = indice_lst[cdr_num-1][0]
    end_indice = indice_lst[cdr_num-1][1]






    # L is the length for the single heavychain. we will merge all the antigen pdb files separately. 

    samples = samples[:,:5,:L,:L]  # SO, b, c , n , n
    #subprocess.call(["mkdir",path_g2])
    #subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    #sample = samples[i][0] # this part is updated on nov 1 2022 due to in saved memory form, there is no distance

    #pdb_id = pdb_name.split("/")[-1][:-4] 

    

    for j in range(len(samples)):
        sample = samples[j]
        # this will only work for inpainting module
    #new_sample = torch.ones([5,128,128])
        path_g2= g_dir+str(sample_index)+"/"
        pre3 = g_dir
        pre = path_g2
        np_sample = sample.detach().numpy()
        seq = get_seq(np_sample)
        #print(seq)
        #c_pdb_with_path(np_sample,seq,path_g2)
        print(np_sample,seq,path_g2,pdb_name,start_indice, end_indice)

        c_pdb_with_path_inpaint(np_sample,seq,path_g2,pdb_name,start_indice, end_indice)
        pre = path_g2 
        cp_pdb(pre+"final_structure.pdb", pre3+"b"+"_"+str(sample_index)+"_"+str(j)+".pdb")
#g1_for_inpaint(0)

import sys

#index = int(sys.argv[1])
f_sample_pkl = sys.argv[1]

ref_pdb_name = sys.argv[2]

sample_index = int(sys.argv[3])

g_all_for_inpaint(f_sample_pkl,ref_pdb_name,sample_index)

#g_all_for_inpaint(index)

