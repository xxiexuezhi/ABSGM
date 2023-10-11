import pickle
import torch
import sys
import numpy as np



padding_lst = [['3IY4_1_H', '9', '120'], ['6GFF_1_H', '6', '124'], ['5DMG_1_H', '11', '119'], ['3QXU_1_H', '4', '126'], ['5BZD_2_H', '1', '128'], ['4OGY_1_H', '6', '123'], ['4OCN_2_H', '6', '124'], ['5CP7_3_H', '10', '120'], ['6NN3_1_H', '4', '125'], ['3QHZ_1_H', '3', '126'], ['5NML_2_H', '10', '120']]






kkk = int(sys.argv[1])
#sigma_max= config.model.sigma_max

start = int(padding_lst[kkk][1])
end = int(padding_lst[kkk][2])


with open("mcmc_testcases11_encoded_dataset_dict_Nov19_heavy_chain_only.pkl","rb") as f:
    test_d= pickle.load(f)


one_test = test_d[kkk]

test_name = one_test["name"]
name_f = test_name.split(".")[0]

target_test = one_test
print(one_test["seq"])
seq = one_test["seq"][start:end]
#print(one_test["data"][0].shape)

print(seq)
a_encode = one_test["data"][:1]
#print(a_encode.shape)



path_g1 = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/sampling/128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022/tmp/"


g_name = path_g1+str(kkk)+"_"+name_f+"_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used_mcmc_on_heavychain_chain_only_nov_dec1_index_all_update_using_seq_sim.pickle"

with open(g_name,"rb") as f:
    samples= pickle.load(f)

print(samples.shape)



def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])



from difflib import SequenceMatcher

#def similar(a, b):
#    return SequenceMatcher(None, a, b).ratio()

#from itertools import zip
def hamming_distance(str1, str2):
    assert len(str1) == len(str2)
    return sum(chr1 != chr2 for chr1, chr2 in zip(str1, str2))

res_lst = []

#from difflib import SequenceMatcher
def similar(a, b):
    return (len(a)-hamming_distance(a,b))/len(a)


distance_ratio= 50
to_save_lst = []
ch = 4 # 0 should be c beta distance. 4 should be seq. 


def cal_distance(one_test,samples):
    min_dis =9999
    test_name = one_test["name"]
    a_encode = one_test["data"][ch:ch+1]
    np_x = samples.detach().numpy()
    for i in range(len(samples)):
        #a_encode = one_test["data"][:1]
        b_encode = samples[i][ch:ch+1]
        distance = np.linalg.norm(a_encode-b_encode)
        seq1 = one_hot_decode(np_x[i][4][:,64-10:64+10])[start:end]
        sim = similar(seq,seq1)
        if sim > 0.5:
            print(i,distance,sim,seq1)
        if distance < min_dis:
            min_dis = distance
        if distance < distance_ratio:
            sam = samples[i].unsqueeze(0)
            to_save_lst.append(sam)
    #saved_lst.append([test_name,min_rmsd, name])
    return min_dis



cal_distance(one_test,samples)
