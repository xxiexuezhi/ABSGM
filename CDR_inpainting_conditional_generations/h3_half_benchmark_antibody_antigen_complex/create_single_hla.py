from Bio import PDB

def c_pdb_chain_lst(pdb_file,chain_ids_to_keep):
    pdb_file = "1i9r.pdb"
    chain_ids_to_keep = ["A", "H", "L"]

    # Create a PDB parser object
    parser = PDB.PDBParser()

    # Load the PDB file into a structure object
    structure = parser.get_structure("pdb", pdb_file)

    # Create a new structure object to hold only the desired chains
    new_structure = PDB.Structure.Structure("new_structure")

    # Loop over the models, chains, and residues in the original structure
    for model in structure:
        for chain in model:
            if chain.id in chain_ids_to_keep:
                # Add the chain to the new structure if its ID is in the list
                new_structure.add(chain)

    # Save the new structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save("new_pdb_file.pdb")


c_pdb_chain_lst("lalal","lalal")
