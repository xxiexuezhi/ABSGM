

length = 128

import numpy as np


def grep_ca_array_from_datasets(d,i):
    bg_cords = d[i]["bb_coords"]
    lst = []
    for i in range(length):
        bg_one = bg_cords[i]
        bg_ca = bg_one[1]
        lst.append(bg_ca)
    return np.array(lst)


def grep_ca_array_from_bb_crds(bb_crds):
    bg_cords = bb_crds
    lst = []
    for i in range(length):
        bg_one = bg_cords[i]
        bg_ca = bg_one[1]
        lst.append(bg_ca)
    return np.array(lst)

import gemmi


def grep_bb_from_pdb(pdb_name):
    p = pdb_name

    st = gemmi.read_structure(p)
    st.setup_entities()
    try:
        polymer = st[0][0].get_polymer()
        #print(polymer)
    except:
        print(f"{p.name} skipped - chain file corrupted")
        return
    if len(polymer) > 128:
        return
    sequence = gemmi.one_letter_code(polymer)
    if "X" in sequence:
        return
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
    return backbone_crds


#bb = grep_bb_from_pdb("training/128_cath_s95_new_config/sampling/structures/iter_350000/10-1-of-16/round_1/structure_before_design.pdb")


import pickle
#f_names = "dataset_antibody_light_heavy_all_info_6000.pkl"

#with open(f_names,"rb") as f:
#    d= pickle.load(f)

#print(grep_ca_array_from_bb_crds(bb).shape)

#print(grep_ca_array_from_datasets(d,0))


import superpose3d

def cal_rmsd(v1,v2):
    rmsd = superpose3d.Superpose3D(v1,v2)[0]
    return rmsd

#g_ca_np = grep_ca_array_from_bb_crds(bb)

#def cal_mini_rmsd(g_ca_np,d):
#    min_rmsd = 999
#    name = ""
#    for i in range(len(d)):
#        coords = grep_ca_array_from_datasets(d,i)
#        if coords==l_coords:
#            rmsd = cal_rmsd(np_t,coords)
#            if rmsd < min_rmsd:
#                min_rmsd=rmsd
#                name = x[i]["seq"]

#    return min_rmsd, name

