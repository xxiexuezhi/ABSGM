
import pickle

with open("mcmc_testcases11_encoded_dataset_dict_Nov19_heavy_chain_only.pkl","rb") as f:
    test_d= pickle.load(f)



name_lst = []


# this is to do MCMC for dis60
aaa = 0
for d in test_d:
    name_f = d["name"].split(".")[0]
    #name = str(aaa)+"_"+name_f+ "_to_save_lst_lst_dis50_save_memory_with_achor_ptrs_used.pickle" #"_to_save_lst_lst_dis40.pickle"  #"_to_save_lst_lst.pickle"  #"_to_save_lst_lst_dis40.pickle"
    name_lst.append(name_f)
    aaa+=1

print(name_lst)

