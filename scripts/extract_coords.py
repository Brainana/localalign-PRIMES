import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

def ca_coordinates_from_pdb(pdb_file, chain_id):
    # extracts the 3D coordinates of the alpha carbons (CA) from the specified chain in a PDB file: output format is np array with shape [N,3]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                ca_coords = []
                for residue in chain:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        ca_coords.append(ca_atom.get_coord())
                return np.array(ca_coords)
    return None

def save_coordinates_as_csv(coordinates, output_file):
    df = pd.DataFrame(coordinates, columns=['X', 'Y', 'Z'])
    df.to_csv(output_file, index=False)

def read_chain_ids(chain_file):
    # mapping between pdb filename and desired chain
    chain_dict = {}
    with open(chain_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pdb_file, chain_id = parts
                chain_dict[pdb_file] = chain_id
    return chain_dict

def process_pdb(pdb_folder, output_folder, chain_file):
    # processes each pdb file in the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chain_dict = read_chain_ids(chain_file)
    all_coordinates = {}

    for pdb_file, chain_id in chain_dict.items():
        pdb_path = os.path.join(pdb_folder, pdb_file)
        if not os.path.exists(pdb_path):
            print(f"Skipping {pdb_file} (File not found).")
            continue
        
        ca_coords = ca_coordinates_from_pdb(pdb_path, chain_id)
        if ca_coords is not None and ca_coords.size > 0:
            print(f"Processing {pdb_file} (Chain {chain_id})... Number of CA atoms: {ca_coords.shape[0]}")
            output_file = os.path.join(output_folder, f"{pdb_file[:-4]}.csv")
            save_coordinates_as_csv(ca_coords, output_file)
            all_coordinates[pdb_file] = ca_coords
        else:
            print(f"Skipping {pdb_file} (No valid CA coordinates found for chain {chain_id}).")

if __name__ == "__main__":
    pdb_folder = "training_data/training_pdbs"
    output_folder = "3d_coords" 
    chain_file = "chain_mapping.txt" 

    process_pdb(pdb_folder, output_folder, chain_file)
