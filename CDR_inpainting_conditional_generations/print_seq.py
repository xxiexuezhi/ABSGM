
from dataset import PaddingCollate,ProteinDataset


name ="protein_dataset_rosetta_benchmark_testset_dataset3_l_with_epitope_match_cdr_4k_all_matching_info_add_after_masking_and_epitope_info_Mar10_no_padding_midptr_64_2023.pkl"


import pickle

with open(name,"rb") as f:
    d1= pickle.load(f)

lst = []

for i in range(len(d1)):
    seq = d1[i]["id"]
    lst.append(seq)

print(lst)
