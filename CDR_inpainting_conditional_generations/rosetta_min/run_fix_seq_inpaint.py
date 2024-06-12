
from pyrosetta.toolbox import mutate_residue

from utils import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.pack.task import operation


import sys,json,math
import tempfile
import numpy as np
from pathlib import Path
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import math
import gemmi
import py3Dmol

from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

pyrosetta.init("-mute all")


vdw_weight = {0: 3.0, 1: 5.0, 2: 10.0}
rsr_dist_weight = {0: 3.0, 1: 2.0, 3: 1.0}
rsr_orient_weight = {0: 1.0, 1: 1.0, 3: 0.5}


import sys,json,math
import tempfile
import numpy as np
from pathlib import Path
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import math
import gemmi
import py3Dmol
from datasets import get_6d_from_pdb

from rosetta_min.utils import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

vdw_weight = {0: 3.0, 1: 5.0, 2: 10.0}
rsr_dist_weight = {0: 3.0, 1: 2.0, 3: 1.0}
rsr_orient_weight = {0: 1.0, 1: 1.0, 3: 0.5}

def init_pyrosetta():
    init_cmd = list()
    init_cmd.append("-multithreading:interaction_graph_threads 1 -multithreading:total_threads 1")
    init_cmd.append("-hb_cen_soft")
    init_cmd.append("-detect_disulf -detect_disulf_tolerance 2.0") # detect disulfide bonds based on Cb-Cb distance (CEN mode) or SG-SG distance (FA mode)
    init_cmd.append("-default_max_cycles 200")
    init_cmd.append("-mute all")
    init(" ".join(init_cmd))

def load_constraints(npz,angle_std,dist_std):
    dist,omega,theta,phi = np.array(npz['dist_abs']), np.array(npz['omega_abs']),np.array(npz['theta_abs']),np.array(npz['phi_abs'])

    dist = dist.astype(np.float32)
    omega = omega.astype(np.float32)
    theta = theta.astype(np.float32)
    phi = phi.astype(np.float32)
    
    L = dist.shape[0]
    
    # dictionary to store Rosetta restraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : []}

    ########################################################
    # dist: 0..20A
    ########################################################
    # Used to filter other restraints
    filter_i, filter_j = np.where(dist>12)
    filter_idx = [(i,j) for i,j in zip(filter_i,filter_j)]
    
    dist = np.triu(dist,1)
    i,j = np.where(dist>0)
    dist = dist[i,j]
    
    for a,b,mean in zip(i,j,dist):
        if (a,b) in filter_idx:
            continue
        harmonic = rosetta.core.scoring.func.HarmonicFunc(mean,dist_std)
        ida = rosetta.core.id.AtomID(5,a+1)
        idb = rosetta.core.id.AtomID(5,b+1)
        rst['dist'].append([a,b,rosetta.core.scoring.constraints.AtomPairConstraint(ida,idb,harmonic)])
        
    print("dist restraints:    %d"%(len(rst['dist'])))


    ########################################################
    # omega: -pi..pi
    ########################################################
    omega = np.triu(omega,1)
    # Use absolute value to not ignore negative values of omega
    i,j = np.where(np.absolute(omega)>0)
    omega = omega[i,j]
    
    for a,b,mean in zip(i,j,omega):
        if (a,b) in filter_idx:
            continue
        harmonic = rosetta.core.scoring.func.CircularHarmonicFunc(mean,np.deg2rad(angle_std))
        id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
        id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
        id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
        id4 = rosetta.core.id.AtomID(2,b+1) # CA-j
        rst['omega'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4,harmonic)])
    print("omega restraints:    %d"%(len(rst['omega'])))


    ########################################################
    # theta: -pi..pi
    ########################################################
    for a in range(L):
        for b in range(L):
            if (a,b) in filter_idx:
                continue
            mean = theta[a][b]
            harmonic = rosetta.core.scoring.func.CircularHarmonicFunc(mean,np.deg2rad(angle_std))
            id1 = rosetta.core.id.AtomID(1,a+1) #  N-i
            id2 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id3 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id4 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['theta'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4,harmonic)])
    print("theta restraints:    %d"%(len(rst['theta'])))


    ########################################################
    # phi: 0..pi
    ########################################################

    for a in range(L):
        for b in range(L):
            if (a,b) in filter_idx:
                continue
            mean = phi[a][b]
            harmonic = rosetta.core.scoring.func.HarmonicFunc(mean,np.deg2rad(angle_std))
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['phi'].append([a,b,rosetta.core.scoring.constraints.AngleConstraint(id1,id2,id3,harmonic)])
    print("phi restraints:    %d"%(len(rst['phi'])))
    
    return rst

def add_rst(pose, rst, sep1, sep2):

    # collect restraints
    array = []
    dist_r = [r for a,b,r in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    omega_r = [r for a,b,r in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    theta_r = [r for a,b,r in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    phi_r   = [r for a,b,r in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2]

    array += dist_r
    array += omega_r
    array += theta_r
    array += phi_r

    if len(array) < 1:
        return

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in array]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)
    
def add_crd_rst(pose, nres, std=1.0, tol=1.0):
    flat_har = rosetta.core.scoring.func.FlatHarmonicFunc(0.0, std, tol)
    rst = list()
    for i in range(1, nres+1):
        xyz = pose.residue(i).atom("CA").xyz() # xyz coord of CA atom
        ida = rosetta.core.id.AtomID(2,i) # CA idx for residue i
        rst.append(rosetta.core.scoring.constraints.CoordinateConstraint(ida, ida, xyz, flat_har)) 

    if len(rst) < 1:
        return
    
    print ("Number of applied coordinate restraints:", len(rst))
    #random.shuffle(rst)

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in rst]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)
    
def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres+1):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)



def run_minimization(
        pdb_name,
        npz,
        seq,
        scriptdir,
        outPath,
        start_indice,
        end_indice,
        pose=None,
        angle_std=10,
        dist_std=2,
        use_fastdesign=True,
        use_fastrelax=True,

):
    L = len(seq)
    rst = load_constraints(npz,angle_std,dist_std)
    scorefxn = create_score_function("ref2015")
    e = 999999
    # make output directory
    outPath.mkdir(exist_ok=True,parents=True)

    sf = ScoreFunction()
    sf.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn.wts')))

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn1.wts')))

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn_vdw.wts')))

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn_cart.wts')))

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.000001, True)
    min_mover_cart.max_iter(300)
    min_mover_cart.cartesian(True)

    ########################################################
    # backbone minimization
    ########################################################
    if False:
        pose0 = pose_from_sequence(seq, 'centroid')
        set_random_dihedral(pose0)
        remove_clash(sf_vdw, min_mover_vdw, pose0)

        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())

        indices_to_design = None

    else:
        pose0 = pose_from_pdb(pdb_name)


        #indices_to_design = [str(i+1) for i,c in enumerate(seq) if c == "_"]
        #indices_to_design = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']
        indices_to_design = [str(k) for k in list(range(start_indice,end_indice))]
        #print(indices_to_design)
#https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.toolbox.mutants.html
        for indice in indices_to_design:
            mutate_residue(pose0,int(indice),seq[int(indice)])



        pose0.dump_pdb("just_after_mutated.pdb")


        to_centroid = SwitchResidueTypeSetMover('centroid')
        to_centroid.apply(pose0)
        #["100,101,102,103,104,105,106,107,108,109]
        prevent_repack = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT() # No repack, no design
        masked_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(",".join(indices_to_design))
        unmasked_residues = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(masked_residues)

        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repack, unmasked_residues))
        # MoveMap
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb_true_range(int(indices_to_design[0]),int(indices_to_design[-1]))
        min_mover1.set_movemap(mm)
        min_mover_vdw.set_movemap(mm)
        min_mover_cart.set_movemap(mm)

    Emin = 999999

    for run in range(10):
        # define repeat_mover here!! (update vdw weights: weak (1.0) -> strong (10.0)
        sf.set_weight(rosetta.core.scoring.vdw, vdw_weight.setdefault(run, 10.0))
        sf.set_weight(rosetta.core.scoring.atom_pair_constraint, rsr_dist_weight.setdefault(run, 1.0))
        sf.set_weight(rosetta.core.scoring.dihedral_constraint, rsr_orient_weight.setdefault(run, 0.5))
        sf.set_weight(rosetta.core.scoring.angle_constraint, rsr_orient_weight.setdefault(run, 0.5))


        min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.001, True)
        min_mover.max_iter(1000)

        if indices_to_design:
            min_mover.set_movemap(mm)

        repeat_mover = RepeatMover(min_mover, 3)

        pose = Pose()
        pose.assign(pose0)
        pose.remove_constraints()

        if run > 0:

            # diversify backbone
            dphi = np.random.uniform(-10,10,L)
            dpsi = np.random.uniform(-10,10,L)

            if indices_to_design:
                for i in indices_to_design:
                    i = int(i)
                    pose.set_phi(i,pose.phi(i)+dphi[i-1])
                    pose.set_psi(i,pose.psi(i)+dpsi[i-1])
            else:
                for i in range(1,L+1):
                    pose.set_phi(i,pose.phi(i)+dphi[i-1])
                    pose.set_psi(i,pose.psi(i)+dpsi[i-1])

            # remove clashes
            remove_clash(sf_vdw, min_mover_vdw, pose)

        # short
        add_rst(pose, rst, 3, 12)
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # medium
        add_rst(pose, rst, 12, 24)
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # long
        add_rst(pose, rst, 24, len(seq))
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # check whether energy has decreased
        E = sf_cart(pose)
        if E < Emin:
            Emin = E
            pose0.assign(pose)

    pose0.remove_constraints()
    pose0.dump_pdb(str(outPath.joinpath("structure_before_design.pdb")))

    if use_fastdesign:
        ############################
        ## sidechain minimization ##
        ############################
        # Convert to all atom representation
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose0)

        # MoveMap
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_chi(True)

        tf.push_back(operation.RestrictToRepacking())

        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        rel_design = pyrosetta.rosetta.protocols.relax.FastRelax()
        rel_design.set_scorefxn(scorefxn)
        rel_design.set_task_factory(tf)
        rel_design.set_movemap(mm)
        rel_design.cartesian(True)
        rel_design.apply(pose0)

        pose0.dump_pdb(str(outPath.joinpath("structure_after_design.pdb")))

#     if use_fastrelax:
#         ########################################################
#         # full-atom refinement
#         ########################################################


#         mmap = MoveMap()
#         #mmap.set_bb_true_range(int(indices_to_design[0]),int(indices_to_design[-1]))
#         #mmap.set_bb(True)
#         mmap.set_chi(True)
#         mmap.set_jump(True)

#         # First round: Repeat 2 torsion space relax w/ strong disto/anglogram constraints
#         sf_fa_round1 = create_score_function('ref2015_cart')
#         sf_fa_round1.set_weight(rosetta.core.scoring.atom_pair_constraint, 3.0)
#         sf_fa_round1.set_weight(rosetta.core.scoring.dihedral_constraint, 1.0)
#         sf_fa_round1.set_weight(rosetta.core.scoring.angle_constraint, 1.0)
#         sf_fa_round1.set_weight(rosetta.core.scoring.pro_close, 0.0)


#         #tf.push_back(operation.RestrictToRepacking())   # Only allow residues to repack. No design at any position.


#         relax_round1 = rosetta.protocols.relax.FastRelax(sf_fa_round1, "%s/data/relax_round1.txt"%scriptdir)

#         relax_round1.constrain_relax_to_native_coords(True)

#         relax_round1.set_movemap(mmap)
#         #relax_round1.set_task_factory(tf)
#         relax_round1.dualspace(True)
#         relax_round1.minimize_bond_angles(True)

#         pose0.remove_constraints()
#         add_rst(pose0, rst, 3, len(seq))
#         try:
#             relax_round1.apply(pose0)
#         except:
#             print("Failed full-atom refinement")

#         # Set options for disulfide tolerance -> 0.5A
#         rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 0.5)

#         sf_fa = create_score_function('ref2015_cart')
#         sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.1)
#         sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
#         sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 0.0)

#         relax_round2 = rosetta.protocols.relax.FastRelax(sf_fa, "%s/data/relax_round2.txt"%scriptdir)
#         relax_round2.set_movemap(mmap)

#         relax_round2.constrain_relax_to_native_coords(True)

#         #relax_round2.set_task_factory(tf)
#         relax_round2.minimize_bond_angles(True)
#         relax_round2.cartesian(True)
#         relax_round2.dualspace(True)

#         pose0.remove_constraints()
#         pose0.conformation().detect_disulfides() # detect disulfide bond again w/ stricter cutoffs
#         # To reduce the number of constraints, only pair distances are considered w/ higher prob cutoffs
#         add_rst(pose0, rst, 3, len(seq))
#         # Instead, apply CA coordinate constraints to prevent drifting away too much (focus on local refinement?)
#         add_crd_rst(pose0, L, std=1.0, tol=2.0)
#         relax_round2.apply(pose0)

        pose0.dump_pdb(str(outPath.joinpath("final_structure.pdb")))
