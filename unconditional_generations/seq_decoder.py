
#path="h3_loop.pickle"
import pickle
path = "dataset_pdb_coord_6d_len80_len128_date_sep3_non_nan.pkl"
with open(path, 'rb') as f:
    d = pickle.load(f)



import numpy as np

def one_hot_decode(seq):
    alphabet = "_ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


x = d[1000]["data"]
np_x = x.detach().numpy()
print(d[1000]["seq"])
print(one_hot_decode(np_x[4][:,64-10:64+11]))
