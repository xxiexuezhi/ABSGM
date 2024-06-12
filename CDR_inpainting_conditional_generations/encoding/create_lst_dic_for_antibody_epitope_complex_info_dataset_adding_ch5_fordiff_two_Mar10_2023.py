import pickle 

with open("processsed_df_table_with_epitope_residue_info.pkl","rb") as f:
    df_grouped= pickle.load(f)


from Bio.PDB.PDBParser import PDBParser

def get_chain_id(pdb_file):

        # Replace 'example.pdf' with the name of your PDF file
    #pdb_file = 'example.pdb'

    # Open the PDB file
    with open(pdb_file) as f:
        for line in f:
            # Check if the line is an ATOM record
            if line.startswith('ATOM'):
                # Extract the chain ID from the line
                chain_id = line[21]
                #print(f"The chain ID is {chain_id}")
                # Exit the loop after the first ATOM record is processed
                return chain_id
        else:
            print("No chain ID found in PDB file")

new_str_lst = []
for i in range(len(df_grouped)):
    new_str_lst.append(''.join(df_grouped["cdr_seq"].values[i]))
df_grouped["cdr_seq_merged"] = new_str_lst


import scipy.spatial

import concurrent.futures
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import scipy
import math
import torch
import gemmi
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from pathlib import Path

#path= Path("/home/xxie92/Desktop/backup_for_dell_15/pdb_projects/code/jupterNotebooks/data/antibody_antigen_antibody_only_data_Mar04_2023/")

#path2 = "/home/xxie92/scratch/rosetta_dock/padding/antigen_antibody_training_data/"
path2 ="/home/xxie92/scratch/igvae/no_pad_pdb/" 
path= Path(path2)
c = 0
lst_names = []
for p in tqdm(path.iterdir()):
    #print(p)
    name = p.stem
    chain = name.split("_")[1]
    lst_names.append(name[:6])



pd_lst = []
epitope_res = []
for index, row in df_grouped.iterrows():
    epitope_res.append(len(row["epitope_tmp_res_lst"]))

    name = row["idcode"]
    chain = row["chainID"]
    pd_lst.append(name+"_"+chain)



import gemmi


def get_seq(p):
    st = gemmi.read_structure(str(p))
    st.setup_entities()
    polymer = st[0][0].get_polymer()
    sequence = gemmi.one_letter_code(polymer)
    return sequence



from abnumber import Chain

# no_index version
def get_cdr_lst(seq):
    lst = []
    try:
        chain = Chain(seq, scheme='chothia')
        lst.append(chain.cdr1_seq)
        lst.append(chain.cdr2_seq)
        lst.append(chain.cdr3_seq)
    except:
        print("not recoginise by Abnumber")
    return lst


def find_row(df,pdb_name,p):
    name = pdb_name.split("_")[0]
    #print(name)
    seq = get_seq(str(p))
    cdr_seq = get_cdr_lst(seq)
    ref_cdr = ''.join(cdr_seq)
    #print(cdr_seq)
    df_filtered = df[(df['idcode'] == name) & (df["cdr_seq_merged"] ==ref_cdr)]
    return  df_filtered


# this is copy from 0827. just to remove _ inside one hot encoding.
import numpy as np
import Bio
from pathlib import Path
import gemmi
from tqdm.notebook import tqdm
import concurrent.futures
import pickle as pkl
import matplotlib.pyplot as plt
import random
import itertools
from tqdm.contrib.concurrent import process_map, thread_map
#from pyrosetta import create_score_function, pose_from_pdb
#import rosetta_min.run as run
#import h5py
#import datasets
import torch
import urllib

import numpy as np
import pandas as pd
import Bio.PDB
from pathlib import Path
import gemmi
from tqdm.notebook import tqdm, trange
import concurrent.futures
import pickle as pkl
import matplotlib.pyplot as plt
import random
import itertools
from tqdm.contrib.concurrent import process_map, thread_map
import torch
import urllib


one_letter_aa_mapping = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}

##### Functions below adapted from trRosetta https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/coords6d.py

# calculate dihedral angles defined by 4 sets of points
def get_dihedrals(a, b, c, d):

    # Ignore divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)
#/home/xxie92/Desktop/backup_for_dell_15/pdb_projects/code/jupterNotebooks/data/antibody_antigen_antibody_only_data_Mar04_2023
# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, dmax=20.0, normalize=True):

    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres),dmax)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    # Normalize all features to [-1,1]
    if normalize:
        # [4A, 20A]
        dist6d = dist6d/dmax*2 - 1
        # [-pi, pi]
        omega6d = omega6d/math.pi
        # [-pi, pi]
        theta6d = theta6d/math.pi
        # [0, pi]
        phi6d = phi6d/math.pi*2 - 1

    return dist6d, omega6d, theta6d, phi6d

##### Process 6D features

def get_dummy_scaler():
    return lambda x:x

def one_hot_encode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(alphabet,range(len(alphabet))))
    seq_idx = [mapping[s] for s in seq]
    return np.eye(len(alphabet))[seq_idx]

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])

def process_6d_coords(seq,crd,size=128):

    seq, crd = pad_seq_and_crd(seq,crd,size)
    n = crd[:,0]
    ca = crd[:,1]
    c = crd[:,2]
    bb = np.array([n,ca,c])
    crd_6d = torch.tensor(np.array(get_coords6d(bb)))

    # Find first and last non-padding residue
    s_stripped = seq.strip("_")
    s_start_aa = s_stripped[0]
    s_end_aa = s_stripped[-1]

    # Find indices of start/end residues
    start_idx = seq.find(s_start_aa)
    end_idx = size-seq[::-1].find(s_end_aa)

    # Set non-padding residues to 0
    ch = torch.ones((size,size))
    ch[start_idx:end_idx,start_idx:end_idx] = 0
    ch = ch.unsqueeze(0)
    crd_6d = torch.cat([crd_6d,ch],dim=0)

    for i in range(4):
        # Boolean matrix of padding residues
        padding = crd_6d[4]==1
        # Distance matrix
        if i == 0:
            crd_6d[i][padding] = 1
        # Phi matrix
        elif i == 3:
            crd_6d[i][padding] = -1
        # Omega, Theta matrix
        else:
            crd_6d[i][padding] = 0

    return crd_6d

def get_6d_from_pdb(path,chain="A",normalize=True):
    st = gemmi.read_structure(str(path))

    # Extract regions defined in - domains
    # get_polymer extracts just the amino acids (excluding H2O, etc.)
    polymer = st[0][chain].get_polymer()
    length = len(list(polymer))

    tmp_res_list = []
    for res in list(polymer):
        res_info = {"name":res.name,"crds":{}}
        res_info["crds"]["CA"] = res.find_atom("CA","*").pos.tolist()
        res_info["crds"]["N"] = res.find_atom("N","*").pos.tolist()
        res_info["crds"]["C"] = res.find_atom("C","*").pos.tolist()
        assert len(res_info["crds"]["C"]) == len(res_info["crds"]["N"]) == len(res_info["crds"]["CA"]) == 3
        tmp_res_list.append(res_info)

    # Concat coordinates
    n = np.array([res["crds"]["N"] for res in tmp_res_list])
    c = np.array([res["crds"]["C"] for res in tmp_res_list])
    ca = np.array([res["crds"]["CA"] for res in tmp_res_list])
    assert n.shape == c.shape == ca.shape

    bb = np.array([n,ca,c])
    return torch.tensor(np.array(get_coords6d(bb,normalize=normalize)))




def extract_6d_coords(path,size=128):
    # Read structure file
    #mmcif_path = path.joinpath(f"{pdb.lower()}.cif")
    mmcif_path = path
    if mmcif_path.is_file():
        st = gemmi.read_structure(str(mmcif_path))
    else:
        #print("1")
        return

    # Extract regions defined in - domains
    # get_polymer extracts just the amino acids (excluding H2O, etc.)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    polymer = st[0][chain].get_polymer()
    length = len(list(polymer))
    #print(list(polymer))

    if length < 40 or length > size:
       # print("2")
        return

    # Parse through each residue and obtain coordinates
    # Try-except catches errors when Ca/N/C coordinates are not present in the PDB file
    try:
        # # Read structure file
        # mmcif_path = path.joinpath(f"{pdb.lower()}.cif")
        # pdb_path = path.joinpath(f"pdb{pdb.lower()}.ent")
        # if mmcif_path.is_file():
        #     st = gemmi.read_structure(str(mmcif_path))
        # elif pdb_path.is_file():
        #     st = gemmi.read_structure(str(pdb_path))
        # else:
        #     return

        # Extract regions defined in - domains
        # get_polymer extracts just the amino acids (excluding H2O, etc.)
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        polymer = st[0][0].get_polymer()
        #polymer = st[0][chain].get_polymer()
        length = len(list(polymer))

        if length < 40 or length > size:
            return

        tmp_res_list = []
        for res in list(polymer):
            res_info = {"name":res.name,"crds":{}}
            res_info["crds"]["CA"] = res.find_atom("CA","*").pos.tolist()
            res_info["crds"]["N"] = res.find_atom("N","*").pos.tolist()
            res_info["crds"]["C"] = res.find_atom("C","*").pos.tolist()
            assert len(res_info["crds"]["C"]) == len(res_info["crds"]["N"]) == len(res_info["crds"]["CA"]) == 3
            tmp_res_list.append(res_info)
    except:
        return

    # Get sequence
    seq = [res["name"] for res in tmp_res_list]
    seq = ''.join([one_letter_aa_mapping[aa] for aa in seq])

    # Concat coordinates
    n = np.array([res["crds"]["N"] for res in tmp_res_list])
    c = np.array([res["crds"]["C"] for res in tmp_res_list])
    ca = np.array([res["crds"]["CA"] for res in tmp_res_list])
    if not n.shape == c.shape == ca.shape:
        return

    # Generate 6D coordinates
    crds = np.stack([n,ca,c])
    crd_6d = torch.tensor(get_coords6d(crds))

    # Pad and add padding channel

    #first change: torch.ones to torch.zeros
    #crd_6d_padded = torch.ones((5,size,size))
    crd_6d_padded = torch.zeros((5,size,size))

    start_idx = (size-length)//2
    end_idx = start_idx + length

    crd_6d_padded[:4,start_idx:end_idx,start_idx:end_idx] = crd_6d

    #second place to change. change it into midlle part for one hot encoding.

    #crd_6d_padded[4,start_idx:end_idx,start_idx:end_idx] = -1
    sequence = gemmi.one_letter_code(polymer)

    mid_point = int(128/2)
    crd_6d_padded[4,start_idx:end_idx,mid_point-10:mid_point+10] = torch.from_numpy(one_hot_encode(seq))
    # Change matrix values of padding residues
    for i in range(4):
        # Boolean matrix of padding residues
        padding = crd_6d_padded[4]==1
        # Distance matrix
        if i == 0:
            crd_6d_padded[i][padding] = 1
        # Phi matrix
        elif i == 3:
            crd_6d_padded[i][padding] = -1
        # Omega, Theta matrix
        else:
            crd_6d_padded[i][padding] = 0

    # Remove NaNs
    if torch.any(torch.isnan(crd_6d_padded)):
        return

    return crd_6d_padded


import torch
def create_ch5(anti_seq,epitope_seq):
    anti_l = len(anti_seq)
    epitope_l = len(epitope_seq)
    total_l = anti_l + epitope_l
    matrix = torch.zeros([total_l,total_l])
    matrix[:anti_l,:anti_l] = 1
    matrix[anti_l:anti_l+epitope_l,:] = -1
    matrix[:,anti_l:anti_l+epitope_l] = -1
    return matrix
    


new_lst = []
def extract_6d_coords_with_epitope(path,tmp_res_list2,size=180):
    # Read structure file
    #mmcif_path = path.joinpath(f"{pdb.lower()}.cif")



    # Extract regions defined in - domains
    # get_polymer extracts just the amino acids (excluding H2O, etc.)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    polymer = st[0][chain].get_polymer()
    length = len(list(polymer))
    #print(list(polymer))

    if length < 40 or length > size:
        print("2")
        return

    # Parse through each residue and obtain coordinates
    # Try-except catches errors when Ca/N/C coordinates are not present in the PDB file
    try:
        # # Read structure file
        # mmcif_path = path.joinpath(f"{pdb.lower()}.cif")
        # pdb_path = path.joinpath(f"pdb{pdb.lower()}.ent")
        # if mmcif_path.is_file():
        #     st = gemmi.read_structure(str(mmcif_path))
        # elif pdb_path.is_file():
        #     st = gemmi.read_structure(str(pdb_path))
        # else:
        #     return

        # Extract regions defined in - domains
        # get_polymer extracts just the amino acids (excluding H2O, etc.)
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        polymer = st[0][0].get_polymer()
        #polymer = st[0][chain].get_polymer()
        length = len(list(polymer))

        if length < 40 or length > size:
            return

        tmp_res_list = []
        for res in list(polymer):
            res_info = {"name":res.name,"crds":{}}
            res_info["crds"]["CA"] = res.find_atom("CA","*").pos.tolist()
            res_info["crds"]["N"] = res.find_atom("N","*").pos.tolist()
            res_info["crds"]["C"] = res.find_atom("C","*").pos.tolist()
            assert len(res_info["crds"]["C"]) == len(res_info["crds"]["N"]) == len(res_info["crds"]["CA"]) == 3
            tmp_res_list.append(res_info)
    except:
        print(3)
        return


   # get seq for antibody and angtigen
    seq1 = [res["name"] for res in tmp_res_list]
    seq2 = [res["name"] for res in tmp_res_list2]

    tmp_res_list3 = tmp_res_list
    tmp_res_list  = tmp_res_list + tmp_res_list2

    

    # Get sequence
    seq = [res["name"] for res in tmp_res_list]
    seq = ''.join([one_letter_aa_mapping[aa] for aa in seq])
    # need to update length in incude epitope info
    length = len(seq)

    # Concat coordinates
    n = np.array([res["crds"]["N"] for res in tmp_res_list])
    c = np.array([res["crds"]["C"] for res in tmp_res_list])
    ca = np.array([res["crds"]["CA"] for res in tmp_res_list])
    if not n.shape == c.shape == ca.shape:
        print(4)
        return

    # Generate 6D coordinates

    crds = np.stack([n,ca,c])
    crd_6d = torch.tensor(get_coords6d(crds))
    # it seems there would be somre wired omega, phi, psi, angles due to the epitopes postions. I will set all NAN into 0.
    crd_6d[torch.isnan(crd_6d)] = 0
    #return crd_6d

    mid_point = 64

    crd_6d_merge = torch.zeros((6,length,length))

    matrix = create_ch5(seq1,seq2)
    crd_6d_merge[:4] = crd_6d
    crd_6d_merge[5] = matrix

    crd_6d_merge[4,:,mid_point-10:mid_point+10] = torch.from_numpy(one_hot_encode(seq))

    

    if torch.any(torch.isnan(crd_6d_merge)):
        #new_lst.append([tmp_res_list,tmp_res_list2])
        print(path,tmp_res_list2)
        print(7777)
        return


    # I dont want to keep doing this padding for the below. So I stoped here.
    return crd_6d_merge

    # Pad and add padding channel

    #first change: torch.ones to torch.zeros
    #crd_6d_padded = torch.ones((5,size,size))
    crd_6d_padded = torch.zeros((5,size,size))

    start_idx = (size-length)//2
    end_idx = start_idx + length

    crd_6d_padded[:4,start_idx:end_idx,start_idx:end_idx] = crd_6d

    #second place to change. change it into midlle part for one hot encoding.

    #crd_6d_padded[4,start_idx:end_idx,start_idx:end_idx] = -1
    # I COMMET OUT THE BLOW LINE DUE TO NEED TO ADDED THE EPITTOPE INFO INTO THE MODEL
    #sequence = gemmi.one_letter_code(polymer)
    sequence = seq
    # the rest is not used
    mid_point = int(180/2)
    crd_6d_padded[4,start_idx:end_idx,mid_point-10:mid_point+10] = torch.from_numpy(one_hot_encode(seq))
    # Change matrix values of padding residues
    for i in range(4):
        # Boolean matrix of padding residues
        padding = crd_6d_padded[4]==1
        # Distance matrix
        if i == 0:
            crd_6d_padded[i][padding] = 1
        # Phi matrix
        elif i == 3:
            crd_6d_padded[i][padding] = -1
        # Omega, Theta matrix
        else:
            crd_6d_padded[i][padding] = 0

    # Remove NaNs
    if torch.any(torch.isnan(crd_6d_padded)):
        print(5)
        return

    return crd_6d_padded


dataset3_l_with_epitope_match_cdr = []

# this is to do a test for fisrt 300.
c = 0
for p in tqdm(path.iterdir()):
    #print(p)
    name = p.stem
    # the below line is commnet out for all antibody chain.
    #if name[:6] not in pd_lst: continue
    #print(name)
    #break
    chain = name.split("_")[1]

    chain = get_chain_id(str(p))
    st = gemmi.read_structure(str(p))
    st.setup_entities()
    try:
        polymer = st[0][0].get_polymer()
        #print(polymer)
    except:
        print(f"{p.name} skipped - chain file corrupted")
        continue
    if len(polymer) > 128:
        continue

    sequence = gemmi.one_letter_code(polymer)
    if "X" in sequence:
        continue
#     try:
        #chain = "A"
    try:
        # comment out the below line and add chain = A
        #chain = "A"
        tmp_res_list2 = []
        #tmp_res_list2 = find_row(df_grouped,name,p).iloc[0]["epitope_tmp_res_lst"]
        #print(tmp_res_list2)
        #print(tmp_res_list2)
        epitope_seq =  [res["name"] for res in tmp_res_list2]
        epitope_seq = ''.join([one_letter_aa_mapping[aa] for aa in epitope_seq])

        encoded_5 = extract_6d_coords_with_epitope(p,tmp_res_list2)
        #print(encoded_5)
    except:
        pass
        print("something wrong here")
        encoded_5 = None
#     except:
#         #chain = "H"
#         print(name)
    backbone_crds = []
    all_atoms_crds = []
    missing=[]
    for idx,res in enumerate(polymer):
        all_atoms = {}
        for atom in res:
            if atom.name == "N":
                n_crd = atom.pos.tolist()
            elif atom.name == "CA":
                ca_crd = atom.pos.tolist()
            elif atom.name == "C":
                c_crd = atom.pos.tolist()
            elif atom.name == "O":
                o_crd = atom.pos.tolist()

            # For chi angle calculations
            all_atoms[atom.name] = atom.pos.tolist()

        # Check if backbone atoms are missing
        if all([i in all_atoms for i in ["N","CA","C","O"]]):
            backbone_crds.append([n_crd,ca_crd,c_crd,o_crd])
        else:
            missing.append(idx)

        # All atoms
        all_atoms_crds.append(all_atoms)
    if encoded_5!=None:
        dataset3_l_with_epitope_match_cdr.append({
            "name":str(p).split("/")[-1],
            "seq":gemmi.one_letter_code(polymer),
            "bb_coords":backbone_crds,
            "all_coords": all_atoms_crds,
            "missing": missing,
            "data":encoded_5,
            "epitope_seq": epitope_seq,
            "epitope_info": tmp_res_list2
        })
    c+=1
    #if c> 100:
    #    break
    ## save as pickle file

