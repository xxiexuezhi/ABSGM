import sys,json,math,os
import tempfile
import numpy as np
from pathlib import Path
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import math
import gemmi
import py3Dmol
import uuid
from pyrosetta import *

import rosetta_min.run_fix_seq as run


#folder_name = sys.argv[2]

#iter_name = "iter_" + folder_name + "0000"

DATASET_PATH = Path.cwd().joinpath(
        "sampling",
        "128_cath_s95_new_config_with_seq"
    )

with open(DATASET_PATH.joinpath("samples.pkl"),"rb") as f:
    samples = pkl.load(f)

with open(DATASET_PATH.joinpath("seq_lst_1000.pkl"),"rb") as f:
    seq_lst = pkl.load(f)


# Extract appropriate batch from samples
TASK_ID=int(sys.argv[1])
BATCH_SIZE=1000
samples = torch.split(torch.tensor(samples),1)[TASK_ID-1].numpy()


#DATASET_NAME = iter_name

# Find length of sequence via mask
for num,sample in enumerate(samples):
    msk = np.round(sample[4])

    L = 128

    #seq = "A"*L
    seq = seq_lst[TASK_ID-1]
    npz={}
    for idx,name in enumerate(["dist","omega","theta","phi"]):
        npz[name] = np.clip(sample[idx].reshape(L,L),-1,1)

    # Inverse scaling
    npz["dist_abs"] = (npz["dist"]+1)*10
    npz["omega_abs"] = npz["omega"]*math.pi
    npz["theta_abs"] = npz["theta"]*math.pi
    npz["phi_abs"] = (npz["phi"]+1)*math.pi/2

    outPath = DATASET_PATH.joinpath(
        "sampling",
        f"{TASK_ID}-{num+1}-of-{BATCH_SIZE}"
    )

    n_iter = 1
    for n in range(n_iter):

        outPath_run = outPath.joinpath(f"round_{n+1}")
        if outPath_run.joinpath("final_structure.pdb").is_file():
            continue

        _ = run.run_minimization(
            npz,
            seq,
            n_iter=n,
            scriptdir=Path.cwd().joinpath("rosetta_min"),
            outPath=outPath_run,
            angle_std=20, # Angular harmonic std
            dist_std=2 # Distance harmonic std
        )

    # Create symlink
    scorefxn_min = create_score_function("ref2015")
    e_min = 9999
    for i in range(n_iter):
        pose = pose_from_pdb(str(outPath.joinpath(f"round_{i+1}","final_structure.pdb")))
        e = scorefxn_min.score(pose)
        if e < e_min:
            best_run = i
            e_min = e

    # Create symlink
    outPath.joinpath(f"best_run").symlink_to(outPath.joinpath(f"round_{best_run+1}").resolve(),target_is_directory=True)

    with open(outPath.joinpath("sample.pkl"),"wb") as f:
        pkl.dump(sample, f)
