kkk = 0

path_g2 = "/home/xxie92/scratch/mcmc_results_11/" + str(kkk) +"/"

import numpy as np

import subprocess

rmsd = 1111
import pickle

#with open("mcmc_testcases11_encoded_dataset_dict_oct13.pkl","rb") as f:
#    test_d= pickle.load(f)



name_lst = []

test_d = []
aaa = 0
for d in test_d:
    name_f = d["name"].split(".")[0]
    name = str(aaa)+"_"+name_f+"_to_save_lst_lst_dis30.pickle"
    name_lst.append(name)
    aaa+=1

print(name_lst)

import torch

sample = torch.randn([2,3])
print(sample)
sample = sample.unsqueeze(0)
print(sample)
lst = [sample,sample,sample]

generated_samples = torch.cat(lst,0)

print(generated_samples.shape)


#with open(path_g2+"min_rmsd/min_rmsd.txt", 'w') as fd:
#    subprocess.call(["echo",str(rmsd)], stdout=fd)


