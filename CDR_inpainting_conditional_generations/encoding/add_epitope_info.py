from abnumber import Chain


#path = "encoded_antigen_antibody_complex_antibody_only_using6d_plus_seq_Mar4_2023.pkl"
#path =  "dataset3_l_with_epitope_match_cdr_4k_all_matching_info.pkl"

#path = "dataset3_l_with_epitope_match_cdr_4k_all_matching_info_update_into_all_data_for_top_left_positions_with_no_padding_Mar10th.pkl"
path ="dataset3_l_with_no_epitope_match_cdr_10k_all_matching_info_update_into_all_data_for_top_left_positions_with_no_padding_Mar10th.pkl"
import pickle

with open(path,"rb") as f:
    d1= pickle.load(f)



import pandas as pd
p2 = 'table2_updated_on_mar6.pkl'
import pickle

#with open(p2,"rb") as f:
#    table2= pickle.load(f)
table2 = []

# Please note, the mid_ptr is updated into 64.

#min_ptr = int(180/2)

min_ptr = int(128/2)



import numpy as np
def one_hot_decode(x):
    np_x = x.detach().numpy()
    seq = np_x[4][:,min_ptr-10:min_ptr+10]
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])

from abnumber import Chain

# no_index version
def get_cdr(seq,i):
    try:
        chain = Chain(seq, scheme='chothia')
        if i ==1:return chain.cdr1_seq
        elif i ==2: return chain.cdr2_seq
        else: return chain.cdr3_seq
    except:
        print(i)

def find_row(df,d,cdr):
    name = d["name"].split("_")[0]
    seq = d["seq"]
    cdr_seq = get_cdr(d["seq"],cdr)
    df_filtered = df[(df['idcode'] == name)    & (df['cdr_seq'] == cdr_seq)]
    return  df_filtered

def one_hot_encode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(alphabet,range(len(alphabet))))
    seq_idx = [mapping[s] for s in seq]
    return np.eye(len(alphabet))[seq_idx]


import torch
d_3_1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
# so the logic is this:

# I need to know the start pos in padded seq.

from abnumber import Chain

def get_pos(cdr,seq):
    return [seq.find(cdr),seq.find(cdr)+len(cdr)]

def get_pos_h(aa_str):
    seq = aa_str
    chain = Chain(seq, scheme='chothia')
    cdr1 = chain.cdr1_seq
    cdr2 = chain.cdr2_seq
    cdr3 = chain.cdr3_seq
    
    re_lst1 = get_pos(cdr1,aa_str)
    re_lst2 = get_pos(cdr2,aa_str)
    re_lst3 = get_pos(cdr3,aa_str)
    lst = [re_lst1,re_lst2,re_lst3]
    h123_str = ','.join([f"{i[0]}:{i[-1]}" for i in lst])
    return h123_str
        

def adding_masking_and_epitopes(d_index):
    #seq = d_index["seq"]
    # Please note, I updated the midpoint to 64.
    padded_seq = one_hot_decode(d_index["data"])

    #for i in range(1,4):
    if False:
        cdr = get_cdr(padded_seq,i)
       # print(cdr)
        res_pair_lst = find_row(table2,d_index,cdr).iloc[0]["process_close_atoms2"]
        start_pos = get_pos(cdr,padded_seq)
        for j in range(len(res_pair_lst)):
            if res_pair_lst[j] != []:
                pos_lst = [-1,1,-2,2]
                for k in range(len(res_pair_lst[j])):
                    if k > 3:
                        break
                    else:
                        if res_pair_lst[j][k] != []:
                            aa = res_pair_lst[j][k][:3]
                           # print(aa)
                            np_aa = one_hot_encode(d_3_1[aa])
                            kkk = pos_lst[k]
                            d_index["data"][4,start_pos[0]+k,54- kkk * 20 : 74 - kkk *20] = torch.tensor(np_aa)
    d_index["pad_seq"] = padded_seq
    d_index["mask_index"] = get_pos_h(padded_seq)
    return d_index

d_index =adding_masking_and_epitopes(d1[60])
print(d_index)


def block(d_index):
    try:
        return adding_masking_and_epitopes(d_index)
    except:
        pass

from tqdm.contrib.concurrent import process_map

data = list(process_map(block, d1, chunksize=10))
#data = []
#for i in range(len(d1)):
#    data.append(block(d1[i]))

# save as pickle file
import pickle
with open("added_cdr123_ss_"+path,"wb") as fout:
    pickle.dump(data,fout)
