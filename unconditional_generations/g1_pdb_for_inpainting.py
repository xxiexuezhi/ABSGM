
#padding_lst = [['4IOF_1_L', '13', '117'], ['3NGB_2_L', '13', '117'], ['3BD4_1_L', '11', '119'], ['6JMR_1_L', '24', '106'], ['4UNT_4_L', '10', '119'], ['5IGX_1_L', '13', '116'], ['6B0N_2_L', '12', '118'], ['5VTA_2_L', '15', '115'], ['3JCX_1_H', '7', '122'], ['5JSA_1_L', '12', '117'], ['6N7U_1_H', '22', '107']]
padding_lst = [['3IY4_1_H', '9', '120'], ['6GFF_1_H', '6', '124'], ['5DMG_1_H', '11', '119'], ['3QXU_1_H', '4', '126'], ['5BZD_2_H', '1', '128'], ['4OGY_1_H', '6', '123'], ['4OCN_2_H', '6', '124'], ['5CP7_3_H', '10', '120'], ['6NN3_1_H', '4', '125'], ['3QHZ_1_H', '3', '126'], ['5NML_2_H', '10', '120']]


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



with open("mcmc_testcases11_encoded_dataset_dict_Nov19_heavy_chain_only.pkl","rb") as f:
    test_d= pickle.load(f)



name_lst = []



saved_lst_name = "_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used_mcmc_on_heavychain_chain_only_nov_dec1_index_all.pickle"
#"_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used_mcmc_on_heavychain_chain_only_nov19.pickle"
# this is to do MCMC for dis60
aaa = 0
for d in test_d:
    name_f = d["name"].split(".")[0]
    name = str(aaa)+"_"+name_f+ saved_lst_name
    #"_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used_mcmc_on_heavychain_chain_only_nov24_index0_2_4_5_6_8.pickle" #"_to_save_lst_lst_dis40.pickle"  #"_to_save_lst_lst.pickle"  #"_to_save_lst_lst_dis40.pickle"_heavychain_chain_only_nov24_index0_2_4_5_6_8
    name_lst.append(name)
    aaa+=1

print(name_lst)

import sys
kkk =int(sys.argv[1])

kkk2 = int(sys.argv[2])

padding_sub_lst = padding_lst[kkk]

start = int(padding_sub_lst[1]) 
end = int(padding_sub_lst[2]) 


n_added = name_lst[kkk]
#n_added = "0_4IOF_1_L_to_save_lst_lst_dis30.pickle"

f_name = "./sampling/128_cath_s95_new_config_with_seq2_ode_heavychain_only_dec_2022/tmp/" +n_added


f_name = "home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/training/inpainting_from_graham/fix_length/samples/iter_120000/sample.pkl"

with open(f_name,"rb") as f:
    samples= pickle.load(f)


#kkk = int(n_added.split("_")[0])



print(test_d[kkk]["name"])

#path_samples_pre = "/home/xxie92/projects/rrg-pmkim/xxie92/diffusion_model/protein-sgm-main/sampling/128_cath_s95_new_config_with_seq2_0916/tmp/"






# dec30 means checking on the generations results regarding dis50 using heavychain only generations. also, I saved all CA coords.

path_g2 = "/home/xxie92/scratch/mcmc_results_11/heavy_only/dis50_save_memory/dec30"# + str(kkk) +"/"
#subprocess.call(["mkdir",path_g2])


import subprocess
subprocess.call(["mkdir",path_g2])
path_g2 = "/home/xxie92/scratch/mcmc_results_11/heavy_only/dis50_save_memory/dec30/"+str(kkk) +"/"#+str(kkk2) + "/"


subprocess.call(["mkdir",path_g2])

path_g2 = "/home/xxie92/scratch/mcmc_results_11/heavy_only/dis50_save_memory/dec30/"+str(kkk) + "/" +str(kkk2) + "/"

subprocess.call(["mkdir",path_g2])

#pre_a = "sampling/"

#after_a = "rename_1k_sample/"



def cp_pdb(pre,after):
    subprocess.call(["cp", pre, after])


#for i in range(1,1001):
#    pre_f = str(i)+"-1-of-1000/round_1/final_structure.pdb"

num = kkk2

a = int(num)
b = int(num+1)

min_rmsd = 10

h3_pos_lst = [[100, 109],
 [102, 115],
 [104, 107],
 [99, 114],
 [96, 116],
 [98, 111],
 [99, 112],
 [104, 108],
 [99, 115],
 [103, 114]]

h3_sub_pos_lst = h3_pos_lst[kkk]
h3_start = h3_sub_pos_lst[0]
h3_end = h3_sub_pos_lst[1]
saved_lst = []

import pickle as pkl


def hack_h3(np_sample,kkk):
    target_sample = numpy.array(test_d[kkk]["data"])
    start = h3_pos_lst[kkk][0]
    end = h3_pos_lst[kkk][1]
    for i in range(5):
        np_sample[i][start:end] = target_sample[i][start:end]
        np_sample[i][:,start:end] = target_sample[i][:,start:end]
    #np_sample[4][:,start:end] = 
    return np_sample
    



for i in range(a,b):
    subprocess.call(["mkdir",path_g2])
    subprocess.call(["mkdir",path_g2+"min_rmsd/"])
    #p = Path(path_g2+"min_rmsd/")
    #p.mkdir(exist_ok=True,parents=True)
    #sample = samples[i][0] # this part is updated on nov 1 2022 due to in saved memory form, there is no distance
    sample = samples[i]
        # this will only work for inpainting module
    new_sample = torch.ones([5,128,128])
    
    for i in range(len(new_sample):
        if i <=3:
            new_sample[i] = sample[i]
        new_sample[4] = sample[7]
    np_sample = new_sample.detach().numpy()
    # this will only work for inpainting module

    seq = get_seq(np_sample)
    # the rest is new updated part on Jan 4th 2023

    print(seq)
    #new_np_sample = hack_h3(np_sample,kkk)
    #seq = get_seq(new_np_sample)
    #np_sample = new_np_sample
    # updated end  here.



    #target_ca =numpy.array(test_d[kkk]["bb_coords"])[:,1] 
    #print(seq)


    for j in range(1):
        c_pdb_with_path(sample,seq,path_g2)
        g_pdb_name = path_g2 +"final_structure.pdb"
       # g_bb = grep_bb_from_pdb(g_pdb_name)
       # g_ca = grep_ca_array_from_bb_crds(g_bb)
       # rmsd=cal_rmsd(target_ca[start:end],g_ca[start:end])
       # print(rmsd)
       # h3_seq = seq[h3_start:h3_end]
       # rmsd_h3 = cal_rmsd(target_ca[h3_start:h3_end],g_ca[h3_start:h3_end])
       # rmsd_h3_m1 = cal_rmsd(target_ca[h3_start:h3_end],g_ca[h3_start-1:h3_end-1])
      #  rmsd_h3_p1 = cal_rmsd(target_ca[h3_start:h3_end],g_ca[h3_start+1:h3_end+1])
       # saved_lst.append([i,seq[start:end],rmsd,h3_seq,rmsd_h3,rmsd_h3_m1,rmsd_h3_p1,seq,g_ca,target_ca])
        # i, generated seq, rmsd (the whole structure without padding),generated h3 sequence,rmsd_h3,rmsd_h3_m1,rmsd_h3_p1)

        #if rmsd < min_rmsd:
            #min_rmsd=rmsd
         #   cp_pdb(g_pdb_name,path_g2+"min_rmsd/"+str(i)+".pdb")
            #with open(path_g2+"min_rmsd/min_rmsd.txt", 'w') as fd:
            #    subprocess.call(["echo",str(rmsd)], stdout=fd)
        
            #with open(path_g2+str(kkk)+"_i_generatedseq_rmsd_h3seq_rmsdh3_rmsdh3m1_rmsdh3p1_ca_coords_g_t_dec30.pickle","wb") as fout:
            #    pkl.dump(saved_lst,fout)

#with open(path_g2+str(kkk)+"_i_generatedseq_rmsd_h3seq_rmsdh3_rmsdh3m1_rmsdh3p1_ca_coords_g_t_dec30.pickle","wb") as fout:
#    pkl.dump(saved_lst,fout)

        #subprocess.call(["echo",str(rmsd),">",path_g2+"min_rmsd/min_rmsd.txt"])






