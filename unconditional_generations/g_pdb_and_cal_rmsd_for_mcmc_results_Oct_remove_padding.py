
padding_lst = [['4IOF_1_L', '13', '117'], ['3NGB_2_L', '13', '117'], ['3BD4_1_L', '11', '119'], ['6JMR_1_L', '24', '106'], ['4UNT_4_L', '10', '119'], ['5IGX_1_L', '13', '116'], ['6B0N_2_L', '12', '118'], ['5VTA_2_L', '15', '115'], ['3JCX_1_H', '7', '122'], ['5JSA_1_L', '12', '117'], ['6N7U_1_H', '22', '107']]



from align_generated_real_structures import grep_bb_from_pdb, grep_ca_array_from_bb_crds, cal_rmsd

from rosetta_do_for_mcmc import c_pdb_with_path

import pickle

import numpy
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



with open("mcmc_testcases11_encoded_dataset_dict_oct13.pkl","rb") as f:
    test_d= pickle.load(f)



name_lst = []


# this is to do MCMC for dis60
aaa = 0
for d in test_d:
    name_f = d["name"].split(".")[0]
    name = str(aaa)+"_"+name_f+ "_to_save_lst_lst_dis40.pickle"  #"_to_save_lst_lst.pickle"  #"_to_save_lst_lst_dis40.pickle"
    name_lst.append(name)
    aaa+=1

print(name_lst)

import sys
kkk =int(sys.argv[1])


padding_sub_lst = padding_lst[kkk]

start = int(padding_sub_lst[1]) -1
end = int(padding_sub_lst[2]) -1


n_added = name_lst[kkk]
#n_added = "0_4IOF_1_L_to_save_lst_lst_dis30.pickle"

f_name = "./sampling/128_cath_s95_new_config_with_seq2_0916/tmp/" +n_added




with open(f_name,"rb") as f:
    samples= pickle.load(f)

#kkk = int(n_added.split("_")[0])



print(test_d[kkk]["name"])

#path_samples_pre = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/sampling/128_cath_s95_new_config_with_seq2_0916/tmp/"

path_g2 = "/home/xxie92/scratch/mcmc_results_11/dis40_3/" + str(kkk) +"/"


import subprocess


#pre_a = "sampling/"

#after_a = "rename_1k_sample/"



def cp_pdb(pre,after):
    subprocess.call(["cp", pre, after])


#for i in range(1,1001):
#    pre_f = str(i)+"-1-of-1000/round_1/final_structure.pdb"


min_rmsd = 20
for i in range(len(samples)):
    subprocess.call(["mkdir",path_g2])
    subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    sample = samples[i][0]
    np_sample = sample.detach().numpy()
    seq = get_seq(np_sample)
    target_ca =numpy.array(test_d[kkk]["bb_coords"])[:,1] 
    print(seq)

    for j in range(1):
        c_pdb_with_path(sample,seq,path_g2)
        g_pdb_name = path_g2 +"final_structure.pdb"
        g_bb = grep_bb_from_pdb(g_pdb_name)
        g_ca = grep_ca_array_from_bb_crds(g_bb)
        rmsd=cal_rmsd(target_ca[start:end],g_ca[start:end])
        print(rmsd)
        if rmsd < min_rmsd:
            min_rmsd=rmsd
            cp_pdb(g_pdb_name,path_g2+"min_rmsd/final_structure.pdb")
            with open(path_g2+"min_rmsd/min_rmsd.txt", 'w') as fd:
                subprocess.call(["echo",str(rmsd)], stdout=fd)

        #subprocess.call(["echo",str(rmsd),">",path_g2+"min_rmsd/min_rmsd.txt"])






