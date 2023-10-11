import pickle
import sys

f_names = "sample.pkl"


pre = "epoch_"

#kkk= sys.argv[1]
f_names = "dataset_antibody_light_heavy_all_info_6000.pkl"
ff_name = "dataset_antibody_light_heavy_all_info_0_4000_date0828_seq_encoded_with_blosum62.pkl"

#with open("mcmc_testcases11_encoded_dataset_dict_oct13.pkl","rb") as f:
#    test_d= pickle.load(f)


with open("training_data_H3_region_only_Jan11_2022.pkl","rb") as f:
    torch_ptr= pickle.load(f)
print(len(torch_ptr))
print(torch_ptr[0]["data"].shape)
