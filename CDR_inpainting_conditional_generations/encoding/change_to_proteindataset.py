# read pickle file
#
import pickle 
#name = "dataset3_l_with_epitope_match_cdr_4k_all_matching_info_without_padding_add_after_masking_and_epitope_info_Mar10_2023.pkl"

path ="dataset3_l_with_no_epitope_match_cdr_10k_all_matching_info_update_into_all_data_for_top_left_positions_with_no_padding_Mar10th.pkl"

name = "added_cdr123_ss_"+path

#with open("dataset3_l_with_epitope_match_cdr_4k_all_matching_info_add_after_masking_and_epitope_info_Mar8_2023.pkl","rb") as f:
with open(name,"rb") as f:
    data= pickle.load(f)

rosetta_name_lst = ['3oz9', '1x9q', '3nps', '3p0y', '2vxv', '4hpy', '2fbj', '3t65',
       '1nlb', '2xwt', '2v17', '1fns', '3lmj', '1dlf', '3g5y', '2e27',
       '3v0w', '2adf', '1oaq', '2d7t', '1mqk', '3ifl', '4h20', '3e8u',
       '2fb4', '3i9g', '3m8o', '3mxw', '1mfa', '3gnm', '3go1', '4nzu',
       '4f57', '4h0h', '3hnt', '3mlr', '1jfq', '2w60', '1gig', '1seq',
       '3hc4', '1mlb', '3liz', '3giz', '3umt', '3eo9', '2ypv', '2r8s',
       '1jpt']

test_lst = []

train_lst = []


for i in range(len(data)):
    try:
        name = data[i]["name"][:4].lower()
        if name in rosetta_name_lst:
            test_lst.append(data[i])

        elif name not in rosetta_name_lst and data[i]["data"]!=None and data[i]["mask_index"]!=None:
            train_lst.append(data[i])
    except:
        pass

# save as pickle file
#import pickle
with open("rosetta_benchmark_"+path,"wb") as fout:
    pickle.dump(test_lst,fout)
#
import pickle
#with open("rosetta_benchmark_test_dataset.pkl","rb") as f:
#    data= pickle.load(f)
#

#train_lst = data

import numpy as np
import scipy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import logging
import warnings
from tqdm.contrib.concurrent import process_map
#import biotite.structure as struc
#from biotite.structure.io.pdb import PDBFile
from torch.utils.data._utils.collate import default_collate

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}


def transform(d):
    r_d = {}
    r_d["id"] = d["name"]
    r_d["coords"] = d["bb_coords"]
    r_d["coords_6d"] = d["data"]
    r_d["aa_str"] = d["pad_seq"]
    r_d["aa"] = [letter_to_num[i] for i in r_d["aa_str"]]
    _,n,n = d["data"].shape
    r_d["mask_pair"] = torch.ones([n,n])
    r_d["ss_indices"] = d["mask_index"]

    return r_d

data = train_lst
structures = data


#from tqdm.contrib.concurrent import process_map

#data = list(process_map(block, d1, chunksize=10))

class ProteinDataset(Dataset):
    def __init__(self, dataset_path="", min_res_num=40, max_res_num=256, ss_constraints=True):
        super().__init__()
        # Ignore biotite warnings
        warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.ss_constraints = ss_constraints

        # Load PDB files into dataset
        # the below means given d.

            # Remove None from self.structures
        #self.structures =list(process_map(lambda: x self.to_tensor(transform(i)), structures, chunksize=10))

        self.structures = [self.to_tensor(transform(i)) for i in structures if i is not None]
    def to_tensor(self, d):
        feat_dtypes = {
            "id": None,
            "coords": torch.float32,
            "coords_6d": torch.float32,
            "aa": torch.long,
            "aa_str": None,
            "mask_pair": torch.bool,
            "ss_indices": None
        }
        for k,v in d.items():
            if feat_dtypes[k] is not None:
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])
        return d

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


proteindata = ProteinDataset()
# save as pickle

# save as pickle file
import pickle
with open("protein_dataset_"+path,"wb") as fout:
    pickle.dump(proteindata,fout)
