


name_lst = ['cut_1i9r_chothia.pdb',
 'cut_4ydl_chothia.pdb',
 'cut_6bp2_chothia.pdb',
 'cut_6nmv_chothia.pdb',
 'cut_6nz7_chothia.pdb',
 'cut_7mep_chothia.pdb',
 'cut_3hmx_chothia.pdb',
 'cut_5kw9_chothia.pdb',
 'cut_6hga_chothia.pdb',
 'cut_6nn3_chothia.pdb',
 'cut_7eng_chothia.pdb',
 'cut_7zfb_chothia.pdb']



from Bio.PDB import PDBParser, PDBIO

# Parse the PDB file
def split_by_chainid(pdb_file):
    name = pdb_file.split(".")[0]
    parser = PDBParser()
    structure = parser.get_structure(name, pdb_file)

    # Split the structure by chain and save each chain as a separate PDB file
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        io = PDBIO()
        io.set_structure(chain)
        io.save(f"{structure.id}_{chain_id}.pdb")


for pdb in name_lst:
    split_by_chainid(pdb)
