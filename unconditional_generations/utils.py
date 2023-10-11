import numpy as np
import gemmi
import py3Dmol
import subprocess
import matplotlib.pyplot as plt

def plot_structure(path:str,chain="A"):
    st = gemmi.read_structure(str(path))
    view = py3Dmol.view(
    data=st.make_minimal_pdb(),
    width=300,
    height=300,
    )
    view.setStyle({"chain": chain}, {"cartoon": {"color": "spectrum"}})

    return view

def calculate_superposition(structure1:gemmi.ResidueSpan, structure2:gemmi.ResidueSpan, rmsd_type="CaP"):
    return gemmi.calculate_superposition(
        structure1,
        structure2,
        structure1.check_polymer_type(),
        getattr(gemmi.SupSelect, rmsd_type)
    )

def run_tmalign(path1,path2,binary_path="tm/TMalign"):
    result = subprocess.run([binary_path,path1,path2],capture_output=True)
    rmsd, tm = result.stdout.decode("UTF-8").split("\n")[12:14]
    #rmsd = rmsd.split(",")[1].split("=")[1].strip()
    tm = tm.split(" ")[1].strip()
    return float(tm)

def structure_from_residues(residues, seqnum_mapping=None):
    chain = gemmi.Chain("A")
    for residue_idx, residue in enumerate(residues):
        residue = residue.clone()
        residue.subchain = "A"
        if seqnum_mapping is not None:
            seqnum = seqnum_mapping[residue_idx]
        else:
            seqnum = residue_idx + 1
        residue.seqid.num = seqnum
        residue.label_seq = seqnum
        chain.add_residue(residue)

    model = gemmi.Model("1")
    model.add_chain(chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    structure.add_entity_types()
    structure.assign_label_seq_id()
    return structure

def save_grid(sample,path=None,axis=0,nrows=4):
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, nrows),
                     axes_pad=0.1,
                     )
    
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    
    for ax, s in zip(grid, sample):
        ax.imshow(s[axis])
    if path:
        plt.savefig(path)

def show_all_channels(sample,path=None,nrows=1):
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
        nrows_ncols=(nrows, 5),
        axes_pad=0.1,
        share_all=True,
    )
    
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    
    grid_idx = 0
    for s in sample:
        for ch in range(5):
            grid[grid_idx].imshow(s[ch])
            grid_idx += 1

    if path:
        plt.savefig(path)