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

import rosetta_min.run as run



DATASET_PATH = Path.cwd().joinpath(
        "training",
        "128_cath_s95_new_config",
        "samples",
        "sample35.pkl"
    )

with open(DATASET_PATH,"rb") as f:
    samples = pkl.load(f)

# Extract appropriate batch from samples
TASK_ID=int(sys.argv[1])
BATCH_SIZE=16
samples = torch.split(torch.tensor(samples),1)[TASK_ID-1].numpy()


DATASET_NAME = "iter_350000"

# Find length of sequence via mask
for num,sample in enumerate(samples):
    msk = np.round(sample[4])

    L = math.sqrt(len(msk[msk==-1]))
    if not (L).is_integer():
        print("Sample skipped due to improper masking...")
        continue
    else:
        L = int(L)

    seq = "A"*L
    npz={}
    for idx,name in enumerate(["dist","omega","theta","phi"]):
        npz[name] = np.clip(sample[idx][msk==-1].reshape(L,L),-1,1)

    # Inverse scaling
    npz["dist_abs"] = (npz["dist"]+1)*10
    npz["omega_abs"] = npz["omega"]*math.pi
    npz["theta_abs"] = npz["theta"]*math.pi
    npz["phi_abs"] = (npz["phi"]+1)*math.pi/2

    outPath = Path.cwd().joinpath(
        "training",
        "128_cath_s95_new_config",
        "sampling",
        "structures",
        DATASET_NAME,
        f"{TASK_ID}-{num+1}-of-{BATCH_SIZE}"
    )

    n_iter = 10
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
