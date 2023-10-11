# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
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
    alphabet = "_ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(alphabet,range(len(alphabet))))
    seq_idx = [mapping[s] for s in seq]
    return np.eye(len(alphabet))[seq_idx]

def one_hot_decode(seq):
    alphabet = "_ARNDCEQGHILKMFPSTWYV"
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
    mmcif_path = path.joinpath(f"{pdb.lower()}.cif")
    if mmcif_path.is_file():
        st = gemmi.read_structure(str(mmcif_path))
    else:
        return

    # Extract regions defined in - domains
    # get_polymer extracts just the amino acids (excluding H2O, etc.)
    polymer = st[0][chain].get_polymer()
    length = len(list(polymer))

    if length < 40 or length > size:
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
    # seq = [res["name"] for res in tmp_res_list]
    # seq = ''.join([one_letter_aa_mapping[aa] for aa in seq])

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
    crd_6d_padded = torch.ones((5,size,size))
    
    start_idx = (size-length)//2
    end_idx = start_idx + length
    
    crd_6d_padded[:4,start_idx:end_idx,start_idx:end_idx] = crd_6d
    crd_6d_padded[4,start_idx:end_idx,start_idx:end_idx] = -1
    
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