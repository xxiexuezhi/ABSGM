import numpy as np
import random
import scipy
from scipy.signal import *
from pyrosetta import *

def init_pyrosetta():
    init_cmd = list()
    init_cmd.append("-multithreading:interaction_graph_threads 1 -multithreading:total_threads 1")
    init_cmd.append("-hb_cen_soft")
    init_cmd.append("-detect_disulf -detect_disulf_tolerance 2.0") # detect disulfide bonds based on Cb-Cb distance (CEN mode) or SG-SG distance (FA mode)
    init_cmd.append("-relax:dualspace true -relax::minimize_bond_angles -default_max_cycles 200")
    init_cmd.append("-mute all")
    init(" ".join(init_cmd))

def set_predicted_dihedral(pose, phi, psi, omega):

    nbins = phi.shape[1]
    bins = np.linspace(-180.,180.,nbins+1)[:-1] + 180./nbins

    nres = pose.total_residue()
    for i in range(nres):
        pose.set_phi(i+1,np.random.choice(bins,p=phi[i]))
        pose.set_psi(i+1,np.random.choice(bins,p=psi[i]))

        if np.random.uniform() < omega[i,0]:
            pose.set_omega(i+1,0)
        else:
            pose.set_omega(i+1,180)

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres+1):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)


#pick phi/psi randomly from:
#-140  153 180 0.135 B
# -72  145 180 0.155 B
#-122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)


def read_fasta(file):
    fasta=""
    first = True
    with open(file, "r") as f:
        for line in f:
            if(line[0] == ">"):
                if first:
                    first = False
                    continue
                else:
                    break
            else:
                line=line.rstrip()
                fasta = fasta + line;
    return fasta


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)

def add_crd_rst(pose, nres, std=1.0, tol=1.0):
    flat_har = rosetta.core.scoring.func.FlatHarmonicFunc(0.0, std, tol)
    rst = list()
    for i in range(1, nres+1):
        xyz = pose.residue(i).atom("CA").xyz() # xyz coord of CA atom
        ida = rosetta.core.id.AtomID(2,i) # CA idx for residue i
        rst.append(rosetta.core.scoring.constraints.CoordinateConstraint(ida, ida, xyz, flat_har)) 

    if len(rst) < 1:
        return
    
    #print ("Number of applied coordinate restraints:", len(rst))
    #random.shuffle(rst)

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in rst]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)

