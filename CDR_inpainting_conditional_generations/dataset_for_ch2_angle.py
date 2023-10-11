import prody as pr
import sidechainnet as scn
from sidechainnet.utils.download import get_resolution_from_pdbid

def process_pdb(filename, pdbid="", include_resolution=False):
    """Return a dictionary containing SidechainNet-relevant data for a given PDB file.

    Args:
        filename (str): Path to existing PDB file.
        pdbid (str): 4-letter string representing the PDB Identifier.
        include_resolution (bool, default=False): If True, query the PDB for the protein
            structure resolution based off of the given pdb_id.

    Returns:
        scndata (dict): A dictionary holding the parsed data attributes of the protein
        structure. Below is a description of the keys:

            The key 'seq' is a 1-letter amino acid sequence.
            The key 'coords' is a (L x NUM_COORDS_PER_RES) x 3 numpy array.
            The key 'angs' is a L x NUM_ANGLES numpy array.
            The key 'is_nonstd' is a L x 1 numpy array with binary values. 1 represents
                that the amino acid at that position was a non-standard amino acid that
                has been modified by SidechainNet into its standard form.
            The key 'unmodified_seq' refers to the original amino acid sequence
                of the protein structure. Some non-standard amino acids are converted into
                their standard form by SidechainNet before measurement. In this case, the
                unmodified_seq variable will contain the original (3-letter code) seq.
            The key 'resolution' is the resolution of the structure as listed on the PDB.
    """
    # First, use Prody to parse the PDB file
    chain = pr.parsePDB(filename)
    # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
    (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
     is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain, replace_nonstd=True)
    scndata = {
        "coords": coords_np,
        "angs": dihedrals_np,
        "seq": observed_sequence,
        "unmodified_seq": unmodified_sequence,
        "is_nonstd": is_nonstd
    }
    # If requested, look up the resolution of the given PDB ID
    if include_resolution:
        assert pdbid, "You must provide a PDB ID to look up the resolution."
        scndata['resolution'] = get_resolution_from_pdbid(pdbid)
    return scndata



import numpy as np
#import perfplot
import math
import numpy as np

def encode_angle_sin_cos(angle):
    return np.sin(angle),np.cos(angle)

def f(x):
    # return math.sqrt(x)
    return encode_angle_sin_cos(x)


vf = np.vectorize(f)


def array_for(x):
    return np.array([f(xi) for xi in x])


def encode_ang_using_pdb(pdb_name):
    a = process_pdb(pdb_name)
    a_angle = a["angs"]
    a_seq = a["seq"]
    a_encoded = array_for(a_angle).transpose([0,2,1])
    a_encoded_reshape = a_encoded.reshape(128,24)
    return a_seq,a_encoded_reshape

def f2(arr12_2): #arr2 size is (12,2) return 12
    # return math.sqrt(x)
    return np.array([np.arctan2(arr2[0],arr2[1]) for arr2 in arr12_2])


#vf = np.vectorize(f)


def array_for2(x):
    return np.array([f2(xi) for xi in x])

# taking the input array size as 128, 24. change it into angles for sidechainnet which is 128, 12
def decode_to_angles(arr):
    arr_reshape = arr.reshape(128,12,2)
    return array_for2(arr_reshape)
    
    #arr_reshape_permute = arr_reshape_permute([0,2,1])
    
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
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from torch.utils.data._utils.collate import default_collate


non_standard_to_standard = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'ASX':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA', # Added ASX => ASP
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYL':'LYS', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS', 'SEC':'CYS', # Added pyrrolysine and selenocysteine
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

lst = []
class ProteinDataset(Dataset):
    def __init__(self, dataset_path, min_res_num=40, max_res_num=256, ss_constraints=True,convert = False):
        super().__init__()
        # Ignore biotite warnings
        warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.ss_constraints = ss_constraints

        # Load PDB files into dataset
        n = 6
        if convert == False:
            paths = list(Path(dataset_path).iterdir())[1000*(n-1):1000* n]
            structures = self.parse_pdb(paths)

        
        # Remove None from self.structures
            self.structures = [self.to_tensor(i) for i in structures if i is not None]
        else:
            self.structures = [i for i in lst] 
    def parse_pdb(self, paths):
        logging.info(f"Processing dataset of length {len(paths)}...")
        data = list(process_map(self.get_features, paths, chunksize=10))
        #print(data)
        return data
    
    def to_tensor(self, d):
        feat_dtypes = {
            "id": None,
            "coords": torch.float32,
            "coords_6d": torch.float32,
            "aa": None,
            "aa_str": None,
            "mask_pair": None,
            "ss_indices": None,
            "angles": None
        }

        for k,v in d.items():
            if feat_dtypes[k] is not None:
                #print("run1")
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])
                #print("run12")

        return d
    def get_features(self, path):
        try:
            with open(path, "r") as f:
                structure = PDBFile.read(f)

            if structure.get_model_count() > 1: return None
            struct = structure.get_structure()
            if struc.get_chain_count(struct) > 1: return None
            _, aa = struc.get_residues(struct)

            # Replace nonstandard amino acids
            for idx,a in enumerate(aa):
                if a not in three_to_one_letter.keys():
                    aa[idx] = non_standard_to_standard.get(a, "UNK")

            one_letter_aa = [three_to_one_letter[i] for i in aa]
            aa_str = ''.join(one_letter_aa)
            aa = [letter_to_num[i] for i in one_letter_aa]
            nres = len(aa)

            if nres > self.max_res_num or nres < self.min_res_num: return None

            a = process_pdb(str(path))
            coords = a["coords"]
            print()
            #return None
            a_angle = a["angs"]
            a_seq = a["seq"]
            seq = a_seq
            a_encoded = array_for(a_angle).transpose([0,2,1])
            a_encoded_reshape = a_encoded.reshape(128,24)
            a_encoded_reshape_pad = torch.zeros([2,128,128])
            mid_ptr = 64

            mid_point = mid_ptr
            a_encoded_reshape_pad[0,:,64-12:64+12] = torch.from_numpy(a_encoded_reshape)
            a_encoded_reshape_pad[1,:,mid_point-10:mid_point+10] = torch.from_numpy(one_hot_encode(seq))
            #print(coords)
            return {
                "id": path.stem,
                "coords":torch.from_numpy(coords),
                "coords_6d": a_encoded_reshape_pad,
                "angles": a["angs"],
                "aa": None,
                "aa_str": a["seq"],
                "mask_pair": None,
                "ss_indices": None # Used for block dropout
            } 
           # print("done")
            
        except:
            #print(coords)
            print(str(path))
            return None
    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]
        
        
